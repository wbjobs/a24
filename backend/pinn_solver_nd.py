import numpy as np
import json
import os
import queue
import threading
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers


class AdaptiveTanh(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.slopes = self.add_weight(
            name='slopes',
            shape=(self.units,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
            regularizer=regularizers.l2(1e-5)
        )
        self.alphas = self.add_weight(
            name='alphas',
            shape=(self.units,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        scaled = tf.multiply(inputs, self.slopes)
        activated = tf.tanh(scaled)
        return tf.multiply(activated, self.alphas)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class FourierFeatureMapping(layers.Layer):
    def __init__(self, num_frequencies=64, scale=10.0,
                 bands=None, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.num_frequencies = num_frequencies
        self.scale = scale
        self.bands = bands if bands is not None else [0.01, 0.1, 1.0, 10.0]
        self.trainable_b = trainable

    def build(self, input_shape):
        in_dim = input_shape[-1]
        total_freqs = 0
        self.B_matrices = []
        for band_idx, band_scale in enumerate(self.bands):
            B = self.add_weight(
                name=f'fourier_B_band{band_idx}',
                shape=(in_dim, self.num_frequencies),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=band_scale
                ),
                trainable=self.trainable_b
            )
            self.B_matrices.append(B)
            total_freqs += self.num_frequencies
        self.output_dim = total_freqs * 2
        super().build(input_shape)

    def call(self, inputs):
        outputs = []
        for B in self.B_matrices:
            projected = tf.matmul(2.0 * np.pi * inputs, B)
            outputs.append(tf.sin(projected))
            outputs.append(tf.cos(projected))
        return tf.concat(outputs, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_frequencies': self.num_frequencies,
            'scale': self.scale,
            'bands': self.bands,
            'trainable': self.trainable_b,
        })
        return config


class MCDropout(layers.Layer):
    def __init__(self, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config


class NDEnvelope(layers.Layer):
    """N维硬约束包络层：支持1D(x), 2D(x,y), 3D(x,y,z)空间 + 时间t"""
    def __init__(self, domain, n_dims=1, ic_func=None, bc_funcs=None, **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        self.n_dims = n_dims
        self.ic_func = ic_func
        self.bc_funcs = bc_funcs or []
        self.spatial_keys = ['x', 'y', 'z'][:n_dims]
        self.t_key = 't'
        self.eps = 1e-8
        self.min_vals = []
        self.max_vals = []
        for k in self.spatial_keys:
            mn, mx = domain[k]
            self.min_vals.append(mn)
            self.max_vals.append(mx)
        self.t_min, self.t_max = domain['t']
        self.ranges = [(mx - mn + self.eps) for mn, mx in zip(self.min_vals, self.max_vals)]
        self.T_range = self.t_max - self.t_min + self.eps

    def _call_ic(self, *spatial_coords):
        n_coords = len(spatial_coords)
        def _fn(*arrs):
            if self.ic_func is None:
                sz = arrs[0].shape[0] if n_coords else 1
                return np.sin(np.pi * arrs[0]).astype(np.float32).reshape(-1, 1)
            arg_dict = {}
            for i, k in enumerate(self.spatial_keys[:n_coords]):
                arg_dict[k] = arrs[i]
            result = self.ic_func(**arg_dict) if callable(self.ic_func) else 0.0
            r = np.asarray(result, dtype=np.float32)
            if r.ndim == 0:
                r = np.full_like(arrs[0], r, dtype=np.float32) if n_coords else np.array([r], dtype=np.float32)
            if r.ndim == 1:
                r = r.reshape(-1, 1)
            return r.astype(np.float32)
        return tf.numpy_function(_fn, list(spatial_coords), tf.float32)

    def call(self, inputs):
        coords, raw_u = inputs
        n_cols = coords.shape[1]
        spatial = []
        for i in range(self.n_dims):
            spatial.append(coords[:, i:i + 1])
        t_coord = coords[:, self.n_dims:self.n_dims + 1]

        d_product = tf.ones_like(t_coord)
        for i in range(self.n_dims):
            mn = self.min_vals[i]
            mx = self.max_vals[i]
            R = self.ranges[i]
            d = 4.0 * (spatial[i] - mn) * (mx - spatial[i]) / (R ** 2)
            d = tf.clip_by_value(d, 0.0, 1.0)
            d_product = d_product * d

        d_t = (t_coord - self.t_min) / self.T_range
        d_t = tf.clip_by_value(d_t, 0.0, 1.0)
        envelope = d_product * d_t

        if self.ic_func is not None:
            u_ic = self._call_ic(*spatial)
            u_ic.set_shape([None, 1])
            t_norm = d_t
            return u_ic * (1.0 - t_norm) + (envelope + t_norm * (1.0 - envelope)) * raw_u
        return envelope * raw_u

    def get_config(self):
        config = super().get_config()
        config.update({'domain': self.domain, 'n_dims': self.n_dims})
        return config


class NDPINNSolver:
    """N维PINN求解器：支持 1D/2D/3D 空间 + 时间，多方程类型（含NS）"""
    EQUATION_TYPES = {
        'heat': '热传导方程',
        'wave': '波动方程',
        'elliptic': '椭圆方程',
        'burgers': 'Burgers方程',
        'ns_incompressible': '不可压缩NS方程',
        'allen_cahn': 'Allen-Cahn方程',
        'kdv': 'KdV方程',
        'reaction_diffusion': '反应扩散方程',
        'sine_gordon': 'Sine-Gordon方程',
        'schrodinger': '非线性薛定谔方程',
        'euler': '欧拉方程',
        'maxwell': '麦克斯韦方程',
        'poisson': '泊松方程',
        'advection': '对流方程',
        'porous_medium': '多孔介质方程',
        'cahn_hilliard': 'Cahn-Hilliard方程',
        'vlasov': 'Vlasov方程',
        'mhd': 'MHD方程',
        'black_scholes': 'Black-Scholes方程',
        'laplace': '拉普拉斯方程',
    }

    OUTPUT_DIMS = {
        'heat': 1, 'wave': 1, 'elliptic': 1, 'burgers': 1,
        'ns_incompressible': 4, 'allen_cahn': 1, 'kdv': 1,
        'reaction_diffusion': 2, 'sine_gordon': 1, 'schrodinger': 2,
        'euler': 3, 'maxwell': 6, 'poisson': 1, 'advection': 1,
        'porous_medium': 1, 'cahn_hilliard': 1, 'vlasov': 1,
        'mhd': 4, 'black_scholes': 1, 'laplace': 1,
    }

    def __init__(self, pde_type='heat', n_dims=1, domain=None, layers=None, params=None,
                 use_fourier=True, use_adaptive_activation=True,
                 use_hard_constraint=True, use_mc_dropout=True,
                 fourier_bands=None, fourier_freqs=32, dropout_rate=0.05):
        self.pde_type = pde_type
        self.n_dims = n_dims
        spatial_keys = ['x', 'y', 'z'][:n_dims]
        default_domain = {}
        for k in spatial_keys:
            default_domain[k] = (0.0, 1.0)
        default_domain['t'] = (0.0, 1.0)
        self.domain = domain or default_domain
        self.spatial_keys = spatial_keys

        n_out = self.OUTPUT_DIMS.get(pde_type, 1)
        if layers is None:
            layers = [n_dims + 1, 128, 128, 128, n_out]
        if layers[-1] != n_out:
            layers[-1] = n_out
        self.layers = layers
        self.params = params or self._default_params(pde_type)
        self.use_fourier = use_fourier
        self.use_adaptive_activation = use_adaptive_activation
        self.use_hard_constraint = use_hard_constraint
        self.use_mc_dropout = use_mc_dropout
        self.fourier_bands = fourier_bands or [0.01, 0.1, 1.0, 10.0]
        self.fourier_freqs = fourier_freqs
        self.dropout_rate = dropout_rate
        self.ic_func = None
        self.bc_funcs = None
        self.model = self._build_network()
        self.history = []
        self.is_trained = False
        self._update_queue = queue.Queue()
        self._update_callback = None

    @staticmethod
    def _default_params(pde_type):
        d = {
            'heat': {'alpha': 0.01},
            'wave': {'c': 1.0},
            'elliptic': {'alpha': 1.0, 'f': 0.0},
            'burgers': {'nu': 0.01},
            'ns_incompressible': {'nu': 0.01, 'rho': 1.0},
            'allen_cahn': {'epsilon': 0.01, 'gamma': 1.0},
            'kdv': {'alpha': 1.0, 'beta': 6.0},
            'reaction_diffusion': {'D1': 0.01, 'D2': 0.01, 'a': 0.1, 'b': 0.1},
            'sine_gordon': {'gamma': 1.0},
            'schrodinger': {'sigma': 1.0},
            'euler': {'gamma': 1.4},
            'maxwell': {'c': 1.0},
            'poisson': {'alpha': 1.0, 'f': 0.0},
            'advection': {'a': 1.0},
            'porous_medium': {'m': 2.0},
            'cahn_hilliard': {'epsilon': 0.01, 'gamma': 1.0},
            'vlasov': {'sigma': 1.0},
            'mhd': {'mu': 0.01, 'eta': 0.01},
            'black_scholes': {'r': 0.05, 'sigma': 0.2, 'K': 100.0},
            'laplace': {},
        }
        return d.get(pde_type, {})

    def _build_network(self):
        input_dim = self.n_dims + 1
        inputs = Input(shape=(input_dim,))
        x_in = inputs

        if self.use_fourier:
            x_in = FourierFeatureMapping(
                num_frequencies=self.fourier_freqs,
                bands=self.fourier_bands,
                trainable=False
            )(x_in)

        hidden_widths = self.layers[1:-1]
        for i, width in enumerate(hidden_widths):
            x_in = layers.Dense(
                width,
                kernel_initializer='glorot_normal',
                kernel_regularizer=regularizers.l2(1e-5)
            )(x_in)

            if self.use_adaptive_activation:
                x_in = AdaptiveTanh(width)(x_in)
            else:
                x_in = layers.Activation('tanh')(x_in)

            if self.use_mc_dropout and i < len(hidden_widths) - 1:
                x_in = MCDropout(rate=self.dropout_rate)(x_in)

        raw_output = layers.Dense(
            self.layers[-1],
            activation=None,
            kernel_initializer='glorot_normal',
            name='raw_u'
        )(x_in)

        if self.use_hard_constraint:
            constrained_output = NDEnvelope(
                domain=self.domain,
                n_dims=self.n_dims,
                ic_func=self.ic_func,
                bc_funcs=self.bc_funcs
            )([inputs, raw_output])
        else:
            constrained_output = raw_output

        return Model(inputs=inputs, outputs=constrained_output, name=f'nd_pinn_{self.pde_type}')

    def _compute_gradients(self, X):
        n = self.n_dims
        spatial = []
        for i in range(n):
            spatial.append(X[:, i:i + 1])
        t = X[:, n:n + 1]

        with tf.GradientTape(persistent=True) as tape2:
            for s in spatial:
                tape2.watch(s)
            tape2.watch(t)
            with tf.GradientTape(persistent=True) as tape1:
                for s in spatial:
                    tape1.watch(s)
                tape1.watch(t)
                u = self.model(tf.concat(spatial + [t], axis=1), training=True)

            first_spatial = []
            for s in spatial:
                first_spatial.append(tape1.gradient(u, s))
            u_t = tape1.gradient(u, t)

            second_spatial = []
            for i, s in enumerate(spatial):
                second_spatial.append(tape2.gradient(first_spatial[i], s))
            u_tt = tape2.gradient(u_t, t)

        del tape1, tape2

        return u, first_spatial, u_t, second_spatial, u_tt

    def _pde_residual(self, X):
        u, d_spatial, u_t, dd_spatial, u_tt = self._compute_gradients(X)
        n = self.n_dims
        P = self.params

        if self.pde_type == 'heat':
            alpha = P.get('alpha', 0.01)
            laplacian = sum(dd_spatial)
            residual = u_t - alpha * laplacian

        elif self.pde_type == 'wave':
            c = P.get('c', 1.0)
            laplacian = sum(dd_spatial)
            residual = u_tt - c ** 2 * laplacian

        elif self.pde_type == 'poisson':
            alpha = P.get('alpha', 1.0)
            f = P.get('f', 0.0)
            laplacian = sum(dd_spatial)
            residual = -alpha * laplacian - f

        elif self.pde_type == 'laplace':
            laplacian = sum(dd_spatial)
            residual = laplacian

        elif self.pde_type == 'advection':
            a = P.get('a', 1.0)
            u_x = d_spatial[0] if n >= 1 else 0.0
            residual = u_t + a * u_x

        elif self.pde_type == 'burgers':
            nu = P.get('nu', 0.01)
            u_x = d_spatial[0] if n >= 1 else 0.0
            u_xx = dd_spatial[0] if n >= 1 else 0.0
            residual = u_t + u * u_x - nu * u_xx

        elif self.pde_type == 'allen_cahn':
            eps = P.get('epsilon', 0.01)
            gamma = P.get('gamma', 1.0)
            laplacian = sum(dd_spatial)
            residual = u_t - eps ** 2 * laplacian + gamma * (u ** 3 - u)

        elif self.pde_type == 'kdv':
            a = P.get('alpha', 1.0)
            b = P.get('beta', 6.0)
            u_x = d_spatial[0] if n >= 1 else 0.0
            u_xx = dd_spatial[0] if n >= 1 else 0.0
            with tf.GradientTape(persistent=True) as _t:
                _t.watch(X[:, 0:1])
                _u = self.model(X, training=True)
                _ux = _t.gradient(_u, X[:, 0:1])
                _uxx = _t.gradient(_ux, X[:, 0:1])
                u_xxx = _t.gradient(_uxx, X[:, 0:1])
            del _t
            residual = u_t + a * u * u_x + b * u_xxx

        elif self.pde_type == 'reaction_diffusion':
            D1 = P.get('D1', 0.01)
            D2 = P.get('D2', 0.01)
            a_p = P.get('a', 0.1)
            b_p = P.get('b', 0.1)
            lap1 = sum([s[:, 0:1] for s in dd_spatial]) if dd_spatial else 0.0
            lap2 = lap1
            u1 = u[:, 0:1]
            u2 = u[:, 1:2] if u.shape[1] > 1 else 0.0 * u
            ut1 = u_t[:, 0:1]
            ut2 = u_t[:, 1:2] if u_t.shape[1] > 1 else 0.0
            r1 = ut1 - D1 * lap1 - a_p + u1 + u1 ** 2 * u2
            r2 = ut2 - D2 * lap2 - b_p - u1 ** 2 * u2
            residual = r1 + r2

        elif self.pde_type == 'sine_gordon':
            gam = P.get('gamma', 1.0)
            laplacian = sum(dd_spatial)
            residual = u_tt - laplacian + gam * tf.sin(u)

        elif self.pde_type == 'ns_incompressible':
            nu = P.get('nu', 0.01)
            rho = P.get('rho', 1.0)
            nd = self.n_dims
            comp = min(nd, 3)
            residuals = []
            for i in range(comp):
                ui = u[:, i:i + 1]
                ui_t = u_t[:, i:i + 1] if u_t.shape[1] > i else 0.0
                conv = 0.0
                diff = 0.0
                for j in range(comp):
                    uj = u[:, j:j + 1]
                    dui_dxj = d_spatial[j][:, i:i + 1] if d_spatial[j].shape[1] > i else 0.0
                    conv = conv + uj * dui_dxj
                    ddui_dxj2 = dd_spatial[j][:, i:i + 1] if dd_spatial[j].shape[1] > i else 0.0
                    diff = diff + ddui_dxj2
                dp_dxi = d_spatial[i][:, comp:comp + 1] if comp < u.shape[1] and d_spatial[i].shape[1] > comp else 0.0
                res_i = rho * (ui_t + conv) + dp_dxi - rho * nu * diff
                residuals.append(res_i)
            div = sum([d_spatial[i][:, i:i + 1] if d_spatial[i].shape[1] > i else 0.0 for i in range(comp)])
            residuals.append(div)
            residual = tf.concat(residuals, axis=1)

        elif self.pde_type == 'black_scholes':
            r = P.get('r', 0.05)
            sig = P.get('sigma', 0.2)
            S = X[:, 0:1]
            V = u
            dV_dS = d_spatial[0]
            d2V_dS2 = dd_spatial[0]
            residual = -u_t + 0.5 * sig ** 2 * S ** 2 * d2V_dS2 + r * S * dV_dS - r * V

        elif self.pde_type == 'cahn_hilliard':
            eps = P.get('epsilon', 0.01)
            gamma = P.get('gamma', 1.0)
            lap = sum(dd_spatial)
            lap_lap = sum(dd_spatial)
            residual = u_t - gamma * eps ** 2 * lap_lap + gamma * lap * (3 * u ** 2 - 1)

        else:
            alpha = P.get('alpha', 0.01)
            laplacian = sum(dd_spatial)
            residual = u_t - alpha * laplacian

        return residual

    def _generate_collocation_points(self, N_f):
        pts = []
        for k in self.spatial_keys:
            mn, mx = self.domain[k]
            pts.append(np.random.uniform(mn, mx, (N_f, 1)).astype(np.float32))
        t_min, t_max = self.domain['t']
        T_f = np.random.uniform(t_min, t_max, (N_f, 1)).astype(np.float32)
        pts.append(T_f)
        return np.hstack(pts)

    def _generate_initial_points(self, N_i, ic_func=None):
        spatial_coords = []
        for k in self.spatial_keys:
            mn, mx = self.domain[k]
            c = np.random.uniform(mn, mx, (N_i, 1)).astype(np.float32)
            spatial_coords.append(c)
        t_min = self.domain['t'][0]
        T_i = np.full((N_i, 1), t_min, dtype=np.float32)

        if ic_func is not None:
            coord_dict = {k: v for k, v in zip(self.spatial_keys, [s[:, 0] for s in spatial_coords])}
            U_i_raw = ic_func(**coord_dict)
        else:
            vals = np.sin(np.pi * spatial_coords[0][:, 0])
            for i in range(1, len(spatial_coords)):
                vals = vals * np.sin(np.pi * spatial_coords[i][:, 0])
            U_i_raw = vals

        U_i = np.asarray(U_i_raw, dtype=np.float32)
        if U_i.ndim == 1:
            U_i = U_i.reshape(-1, 1)
        if U_i.shape[1] < self.layers[-1]:
            pad = np.zeros((U_i.shape[0], self.layers[-1] - U_i.shape[1]), dtype=np.float32)
            U_i = np.hstack([U_i, pad])
        elif U_i.shape[1] > self.layers[-1]:
            U_i = U_i[:, :self.layers[-1]]

        all_coords = spatial_coords + [T_i]
        return np.hstack(all_coords), U_i.astype(np.float32)

    def _generate_boundary_points(self, N_b, bc_funcs=None):
        all_boundary_X = []
        all_boundary_U = []
        t_min, t_max = self.domain['t']
        T = np.random.uniform(t_min, t_max, (N_b, 1)).astype(np.float32)

        for dim_idx, k in enumerate(self.spatial_keys):
            mn, mx = self.domain[k]
            for side, val in enumerate([mn, mx]):
                coords = []
                for j, sk in enumerate(self.spatial_keys):
                    if j == dim_idx:
                        c = np.full((N_b, 1), val, dtype=np.float32)
                    else:
                        mnj, mxj = self.domain[sk]
                        c = np.random.uniform(mnj, mxj, (N_b, 1)).astype(np.float32)
                    coords.append(c)
                coords.append(T)

                u_vals = np.zeros((N_b, self.layers[-1]), dtype=np.float32)
                if bc_funcs is not None and len(bc_funcs) > dim_idx * 2 + side:
                    fn = bc_funcs[dim_idx * 2 + side]
                    if fn is not None:
                        coord_dict = {sk: coords[j][:, 0] for j, sk in enumerate(self.spatial_keys)}
                        coord_dict['t'] = T[:, 0]
                        r = fn(**coord_dict)
                        r_arr = np.asarray(r, dtype=np.float32)
                        if r_arr.ndim == 0:
                            r_arr = np.full((N_b,), r_arr, dtype=np.float32)
                        if r_arr.ndim == 1:
                            r_arr = r_arr.reshape(-1, 1)
                        if r_arr.shape[1] < self.layers[-1]:
                            pad = np.zeros((N_b, self.layers[-1] - r_arr.shape[1]), dtype=np.float32)
                            r_arr = np.hstack([r_arr, pad])
                        u_vals = r_arr[:, :self.layers[-1]].astype(np.float32)

                all_boundary_X.append(np.hstack(coords))
                all_boundary_U.append(u_vals)

        return np.vstack(all_boundary_X), np.vstack(all_boundary_U).astype(np.float32)

    def set_update_callback(self, callback_fn):
        self._update_callback = callback_fn

    def train(self, ic_func=None, bc_funcs=None,
              N_f=10000, N_i=500, N_b=500,
              epochs=5000, learning_rate=1e-3, verbose=True,
              lr_decay=True, log_interval=50,
              hvd=None):
        self.ic_func = ic_func
        self.bc_funcs = bc_funcs

        if self.use_hard_constraint:
            self.model = self._build_network()

        X_f_np = self._generate_collocation_points(N_f)
        X_f = tf.constant(X_f_np, dtype=tf.float32)
        X_i_np, U_i_np = self._generate_initial_points(N_i, ic_func)
        X_i = tf.constant(X_i_np, dtype=tf.float32)
        U_i = tf.constant(U_i_np, dtype=tf.float32)
        X_b_np, U_b_np = self._generate_boundary_points(N_b, bc_funcs)
        X_b = tf.constant(X_b_np, dtype=tf.float32)
        U_b = tf.constant(U_b_np, dtype=tf.float32)

        total_steps = epochs
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_steps,
            alpha=0.01
        ) if lr_decay else learning_rate

        if hvd is not None:
            try:
                import horovod.tensorflow as hvd_lib
                opt = tf.keras.optimizers.Adam(learning_rate=learning_rate * hvd_lib.size())
                opt = hvd_lib.DistributedOptimizer(opt)
            except Exception:
                opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        optimizer = opt

        self.history = []

        @tf.function
        def train_step(X_f_tf, X_i_tf, U_i_tf, X_b_tf, U_b_tf):
            with tf.GradientTape() as tape:
                f_pred = self._pde_residual(X_f_tf)
                loss_f = tf.reduce_mean(tf.square(f_pred))

                if not self.use_hard_constraint:
                    u_pred_i = self.model(X_i_tf, training=True)
                    loss_i = tf.reduce_mean(tf.square(u_pred_i - U_i_tf))
                    u_pred_b = self.model(X_b_tf, training=True)
                    loss_b = tf.reduce_mean(tf.square(u_pred_b - U_b_tf))
                else:
                    loss_i = tf.constant(0.0, dtype=tf.float32)
                    loss_b = tf.constant(0.0, dtype=tf.float32)

                reg_loss = tf.add_n(self.model.losses) if self.model.losses else tf.constant(0.0, dtype=tf.float32)
                loss = loss_f + loss_i + loss_b + 1e-4 * reg_loss

            if hvd is not None:
                try:
                    import horovod.tensorflow as hvd_lib
                    tape = hvd_lib.DistributedGradientTape(tape)
                except Exception:
                    pass

            grads = tape.gradient(loss, self.model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss, loss_i, loss_b, loss_f

        start_epoch = 0
        is_rank0 = (hvd is None) or (hvd.rank() == 0 if hasattr(hvd, 'rank') else True)

        for epoch in range(epochs):
            try:
                loss, loss_i, loss_b, loss_f = train_step(X_f, X_i, U_i, X_b, U_b)
            except Exception as e:
                if is_rank0:
                    print(f'tf.function failed at epoch {epoch}, falling back to eager: {e}')
                with tf.GradientTape() as tape:
                    f_pred = self._pde_residual(X_f)
                    loss_f = tf.reduce_mean(tf.square(f_pred))
                    if not self.use_hard_constraint:
                        u_pred_i = self.model(X_i, training=True)
                        loss_i = tf.reduce_mean(tf.square(u_pred_i - U_i))
                        u_pred_b = self.model(X_b, training=True)
                        loss_b = tf.reduce_mean(tf.square(u_pred_b - U_b))
                    else:
                        loss_i = tf.constant(0.0)
                        loss_b = tf.constant(0.0)
                    loss = loss_f + loss_i + loss_b
                if hvd is not None:
                    try:
                        import horovod.tensorflow as hvd_lib
                        tape = hvd_lib.DistributedGradientTape(tape)
                    except Exception:
                        pass
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                loss = float(loss.numpy()) if hasattr(loss, 'numpy') else float(loss)
                loss_i = float(loss_i.numpy()) if hasattr(loss_i, 'numpy') else 0.0
                loss_b = float(loss_b.numpy()) if hasattr(loss_b, 'numpy') else 0.0
                loss_f = float(loss_f.numpy()) if hasattr(loss_f, 'numpy') else 0.0

            loss_val = float(loss) if not hasattr(loss, 'numpy') else float(loss.numpy())
            loss_i_val = float(loss_i) if not hasattr(loss_i, 'numpy') else float(loss_i.numpy())
            loss_b_val = float(loss_b) if not hasattr(loss_b, 'numpy') else float(loss_b.numpy())
            loss_f_val = float(loss_f) if not hasattr(loss_f, 'numpy') else float(loss_f.numpy())

            record = {
                'epoch': epoch + 1,
                'loss': loss_val,
                'loss_ic': loss_i_val,
                'loss_bc': loss_b_val,
                'loss_pde': loss_f_val,
            }
            self.history.append(record)

            if is_rank0 and self._update_callback is not None and (epoch + 1) % log_interval == 0:
                try:
                    self._update_callback(record)
                except Exception:
                    pass

            if is_rank0 and verbose and (epoch + 1) % max(log_interval * 5, 100) == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss_val:.3e} | PDE: {loss_f_val:.3e}")

        self.is_trained = True
        return self.history

    def predict(self, X, mc_samples=1):
        X = np.asarray(X, dtype=np.float32)
        if mc_samples <= 1 or not self.use_mc_dropout:
            return self.model(X, training=False).numpy()
        preds = []
        for _ in range(mc_samples):
            preds.append(self.model(X, training=True).numpy())
        preds = np.stack(preds, axis=0)
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        return mean, std

    def predict_grid(self, grid_sizes=None, mc_samples=30):
        """多维度网格预测，grid_sizes={'x':nx,'y':ny,'z':nz,'t':nt}"""
        if grid_sizes is None:
            grid_sizes = {'t': 50}
            for k in self.spatial_keys:
                grid_sizes[k] = 50

        axes = []
        for k in self.spatial_keys:
            mn, mx = self.domain[k]
            sz = grid_sizes.get(k, 50)
            axes.append(np.linspace(mn, mx, sz).astype(np.float32))
        t_min, t_max = self.domain['t']
        t_ax = np.linspace(t_min, t_max, grid_sizes.get('t', 50)).astype(np.float32)
        all_axes = axes + [t_ax]

        meshes = np.meshgrid(*all_axes, indexing='ij')
        flat_list = [m.reshape(-1, 1) for m in meshes]
        X_flat = np.hstack(flat_list).astype(np.float32)

        use_mc = mc_samples > 1 and self.use_mc_dropout
        if use_mc:
            U_mean_flat, U_std_flat = self.predict(X_flat, mc_samples=mc_samples)
            has_unc = True
        else:
            U_mean_flat = self.predict(X_flat, mc_samples=1)
            has_unc = False

        n_out = self.layers[-1]
        mesh_shape = tuple([len(a) for a in all_axes])
        result = {
            'axes': {},
            'shape': list(mesh_shape),
            'n_dims': self.n_dims,
            'n_outputs': n_out,
            'has_uncertainty': has_unc,
            'output_names': self._output_names(),
        }
        for i, k in enumerate(self.spatial_keys):
            result['axes'][k] = axes[i].tolist()
        result['axes']['t'] = t_ax.tolist()

        for o in range(n_out):
            u_mean = U_mean_flat[:, o].reshape(mesh_shape)
            result[f'u_{o}'] = u_mean.tolist()
            if has_unc:
                u_std = U_std_flat[:, o].reshape(mesh_shape)
                result[f'u_std_{o}'] = u_std.tolist()
                result[f'u_unc_{o}'] = (2.0 * u_std).tolist()

        return result

    def _output_names(self):
        names_map = {
            'ns_incompressible': ['u', 'v', 'w', 'p'],
            'reaction_diffusion': ['u', 'v'],
            'schrodinger': ['Re(psi)', 'Im(psi)'],
            'euler': ['rho', 'u', 'E'],
            'maxwell': ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
            'mhd': ['rho', 'u', 'B', 'p'],
        }
        base = names_map.get(self.pde_type, ['u'])
        while len(base) < self.layers[-1]:
            base.append(f'out{len(base)}')
        return base[:self.layers[-1]]

    def predict_with_uncertainty(self, grid_sizes=None, mc_samples=30):
        return self.predict_grid(grid_sizes, mc_samples)

    def get_weights(self):
        return [{'data': w.tolist(), 'shape': list(w.shape)} for w in self.model.get_weights()]

    def set_weights(self, weights_data):
        self.model.set_weights([np.array(w['data']).reshape(w['shape']) for w in weights_data])
        self.is_trained = True


def create_ic_func_nd(expression_str, params=None, spatial_keys=None):
    """多维IC/BC函数工厂"""
    params = params or {}
    spatial_keys = spatial_keys or ['x']

    def ic_func(**kwargs):
        safe_dict = {
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
            'pi': np.pi, 'abs': np.abs, 'sqrt': np.sqrt,
            'tanh': np.tanh, 'sinh': np.sinh, 'cosh': np.cosh,
            'log': np.log,
            **{k: np.float32(v) for k, v in params.items()},
            **{k: np.asarray(v, dtype=np.float32) for k, v in kwargs.items()},
        }
        result = eval(expression_str, {"__builtins__": {}}, safe_dict)
        return np.asarray(result, dtype=np.float32)

    return ic_func


class PINNSolver(NDPINNSolver):
    """兼容1D + t老代码的别名"""
    def __init__(self, *args, **kwargs):
        kwargs['n_dims'] = 1
        if 'domain' in kwargs:
            d = kwargs['domain']
            if isinstance(d, dict) and 'x' in d and 't' in d:
                pass
            elif isinstance(d, dict) and len(d) == 2:
                pass
        super().__init__(*args, **kwargs)

    def predict_grid(self, nx=100, nt=100, mc_samples=30):
        """兼容旧1D+t API"""
        result = super().predict_grid({'x': nx, 't': nt}, mc_samples)
        x = np.array(result['axes']['x'])
        t = np.array(result['axes']['t'])
        U = np.array(result['u_0'])
        perm = list(range(1, U.ndim)) + [0]
        U_tfirst = np.transpose(U, (1, 0))
        if result['has_uncertainty']:
            U_std = np.transpose(np.array(result['u_std_0']), (1, 0))
            U_unc = np.transpose(np.array(result['u_unc_0']), (1, 0))
            return x, t, U_tfirst, U_std, U_unc
        return x, t, U_tfirst
