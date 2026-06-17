import numpy as np
import json
import os
import queue
import threading
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
        self.bands = bands if bands is not None else [0.01, 0.1, 1.0, 10.0, 100.0]
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


class HardConstraintEnvelope(layers.Layer):
    def __init__(self, domain, ic_func=None, bc_left_func=None, bc_right_func=None, **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        self.ic_func = ic_func
        self.bc_left_func = bc_left_func
        self.bc_right_func = bc_right_func
        self.x_min, self.x_max = domain['x']
        self.t_min, self.t_max = domain['t']
        self.eps = 1e-8
        self.L = self.x_max - self.x_min + self.eps
        self.T = self.t_max - self.t_min + self.eps

    def _call_ic(self, x_tensor):
        def _fn(x_np):
            if self.ic_func is None:
                return np.sin(np.pi * x_np).astype(np.float32).reshape(-1, 1)
            r = self.ic_func(x_np)
            r = np.asarray(r, dtype=np.float32)
            if r.ndim == 0:
                r = np.full_like(x_np, r, dtype=np.float32)
            if r.ndim == 1:
                r = r.reshape(-1, 1)
            return r.astype(np.float32)
        return tf.numpy_function(_fn, [x_tensor], tf.float32)

    def _call_bc_left(self, t_tensor):
        def _fn(t_np):
            if self.bc_left_func is None:
                return np.zeros_like(t_np, dtype=np.float32).reshape(-1, 1)
            r = self.bc_left_func(t_np)
            r = np.asarray(r, dtype=np.float32)
            if r.ndim == 0:
                r = np.full_like(t_np, r, dtype=np.float32)
            if r.ndim == 1:
                r = r.reshape(-1, 1)
            return r.astype(np.float32)
        return tf.numpy_function(_fn, [t_tensor], tf.float32)

    def _call_bc_right(self, t_tensor):
        def _fn(t_np):
            if self.bc_right_func is None:
                return np.zeros_like(t_np, dtype=np.float32).reshape(-1, 1)
            r = self.bc_right_func(t_np)
            r = np.asarray(r, dtype=np.float32)
            if r.ndim == 0:
                r = np.full_like(t_np, r, dtype=np.float32)
            if r.ndim == 1:
                r = r.reshape(-1, 1)
            return r.astype(np.float32)
        return tf.numpy_function(_fn, [t_tensor], tf.float32)

    def call(self, inputs):
        x_t, raw_u = inputs
        x_coords = x_t[:, 0:1]
        t_coords = x_t[:, 1:2]

        d_bc = 4.0 * (x_coords - self.x_min) * (self.x_max - x_coords) / (self.L ** 2)
        d_ic = (t_coords - self.t_min) / self.T
        d_ic = tf.clip_by_value(d_ic, 0.0, 1.0)
        d_bc = tf.clip_by_value(d_bc, 0.0, 1.0)
        envelope = d_bc * d_ic

        if self.ic_func is not None and self.bc_left_func is not None and self.bc_right_func is not None:
            u_ic = self._call_ic(x_coords)
            u_b0 = self._call_bc_left(t_coords)
            u_b1 = self._call_bc_right(t_coords)
            u_ic.set_shape([None, 1])
            u_b0.set_shape([None, 1])
            u_b1.set_shape([None, 1])

            x_norm = (x_coords - self.x_min) / self.L
            t_norm = (t_coords - self.t_min) / self.T
            bc_linear = u_b0 * (1.0 - x_norm) + u_b1 * x_norm
            ic_bc_combined = u_ic * (1.0 - t_norm) + bc_linear * t_norm

            return ic_bc_combined + envelope * raw_u
        else:
            return envelope * raw_u

    def get_config(self):
        config = super().get_config()
        config.update({'domain': self.domain})
        return config


class HardConstraintPINN:
    def __init__(self, pde_type='heat', domain=None, layers=None, params=None,
                 use_fourier=True, use_adaptive_activation=True,
                 use_hard_constraint=True, use_mc_dropout=True,
                 fourier_bands=None, fourier_freqs=32,
                 dropout_rate=0.05):
        self.pde_type = pde_type
        self.domain = domain or {'x': (0.0, 1.0), 't': (0.0, 1.0)}
        self.layers = layers or [2, 128, 128, 128, 128, 1]
        self.params = params or {'alpha': 0.01}
        self.use_fourier = use_fourier
        self.use_adaptive_activation = use_adaptive_activation
        self.use_hard_constraint = use_hard_constraint
        self.use_mc_dropout = use_mc_dropout
        self.fourier_bands = fourier_bands or [0.01, 0.1, 1.0, 10.0]
        self.fourier_freqs = fourier_freqs
        self.dropout_rate = dropout_rate
        self.ic_func = None
        self.bc_left_func = None
        self.bc_right_func = None
        self.model = self._build_network()
        self.history = []
        self.is_trained = False
        self._update_queue = queue.Queue()
        self._update_callback = None

    def _distance_to_boundary(self, x, t):
        x_min, x_max = self.domain['x']
        t_min, t_max = self.domain['t']
        x_l = x - x_min
        x_r = x_max - x
        t_init = t - t_min
        eps = 1e-8
        d_bc = (x_l * x_r) / ((x_max - x_min) ** 2 / 4.0 + eps)
        d_ic = t_init / (t_max - t_min + eps)
        return d_bc, d_ic

    def _build_network(self):
        inputs = Input(shape=(2,))
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
            constrained_output = HardConstraintEnvelope(
                domain=self.domain,
                ic_func=self.ic_func,
                bc_left_func=self.bc_left_func,
                bc_right_func=self.bc_right_func
            )([inputs, raw_output])
        else:
            constrained_output = raw_output

        model = Model(inputs=inputs, outputs=constrained_output, name='hard_constraint_pinn')
        return model

    def _compute_gradients(self, X):
        x = X[:, 0:1]
        t = X[:, 1:2]
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            tape2.watch(t)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x)
                tape1.watch(t)
                u = self.model(tf.concat([x, t], axis=1), training=True)
            u_x = tape1.gradient(u, x)
            u_t = tape1.gradient(u, t)
        u_xx = tape2.gradient(u_x, x)
        u_tt = tape2.gradient(tape1.gradient(u, t), t)
        del tape1, tape2
        return u, u_x, u_t, u_xx, u_tt

    def _pde_residual(self, X):
        u, u_x, u_t, u_xx, u_tt = self._compute_gradients(X)
        alpha = self.params.get('alpha', 0.01)
        if self.pde_type == 'heat':
            residual = u_t - alpha * u_xx
        elif self.pde_type == 'wave':
            c = self.params.get('c', 1.0)
            residual = u_tt - c ** 2 * u_xx
        elif self.pde_type == 'elliptic':
            f = self.params.get('f', 0.0)
            residual = -alpha * u_xx - f
        else:
            residual = u_t - alpha * u_xx
        return residual

    def _generate_collocation_points(self, N_f):
        x_min, x_max = self.domain['x']
        t_min, t_max = self.domain['t']
        X_f = np.random.uniform(x_min, x_max, (N_f, 1)).astype(np.float32)
        T_f = np.random.uniform(t_min, t_max, (N_f, 1)).astype(np.float32)
        return np.hstack([X_f, T_f])

    def _generate_initial_points(self, N_i, ic_func=None):
        x_min, x_max = self.domain['x']
        t_min = self.domain['t'][0]
        X_i = np.linspace(x_min, x_max, N_i).astype(np.float32).reshape(-1, 1)
        T_i = np.full((N_i, 1), t_min, dtype=np.float32)
        if ic_func is not None:
            U_i = ic_func(X_i)
        else:
            U_i = np.sin(np.pi * X_i).astype(np.float32).reshape(-1, 1)
        return np.hstack([X_i, T_i]), np.asarray(U_i, dtype=np.float32)

    def _generate_boundary_points(self, N_b, bc_funcs=None):
        x_min, x_max = self.domain['x']
        t_min, t_max = self.domain['t']
        T_b = np.linspace(t_min, t_max, N_b).astype(np.float32).reshape(-1, 1)
        X_left = np.full((N_b, 1), x_min, dtype=np.float32)
        X_right = np.full((N_b, 1), x_max, dtype=np.float32)
        X_bnd = np.vstack([X_left, X_right])
        T_bnd = np.vstack([T_b, T_b])
        if bc_funcs is not None:
            U_left = np.asarray(bc_funcs[0](T_b), dtype=np.float32)
            U_right = np.asarray(bc_funcs[1](T_b), dtype=np.float32)
            U_bnd = np.vstack([U_left, U_right])
        else:
            U_bnd = np.zeros((2 * N_b, 1), dtype=np.float32)
        return np.hstack([X_bnd, T_bnd]), U_bnd

    def set_update_callback(self, callback_fn):
        self._update_callback = callback_fn

    def train(self, ic_func=None, bc_funcs=None,
              N_f=10000, N_i=500, N_b=500,
              epochs=5000, learning_rate=1e-3, verbose=True,
              lr_decay=True, log_interval=50):
        self.ic_func = ic_func
        if bc_funcs is not None:
            self.bc_left_func, self.bc_right_func = bc_funcs

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

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
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

            grads = tape.gradient(loss, self.model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            return loss, loss_i, loss_b, loss_f

        for epoch in range(epochs):
            try:
                loss, loss_i, loss_b, loss_f = train_step(X_f, X_i, U_i, X_b, U_b)
            except Exception as e:
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
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                loss = loss.numpy().item()
                loss_i = loss_i.numpy().item() if hasattr(loss_i, 'numpy') else 0.0
                loss_b = loss_b.numpy().item() if hasattr(loss_b, 'numpy') else 0.0
                loss_f = loss_f.numpy().item() if hasattr(loss_f, 'numpy') else 0.0

            loss_val = float(loss) if not hasattr(loss, 'numpy') else loss.numpy().item()
            loss_i_val = float(loss_i) if not hasattr(loss_i, 'numpy') else loss_i.numpy().item()
            loss_b_val = float(loss_b) if not hasattr(loss_b, 'numpy') else loss_b.numpy().item()
            loss_f_val = float(loss_f) if not hasattr(loss_f, 'numpy') else loss_f.numpy().item()

            record = {
                'epoch': epoch + 1,
                'loss': loss_val,
                'loss_ic': loss_i_val,
                'loss_bc': loss_b_val,
                'loss_pde': loss_f_val,
            }
            self.history.append(record)

            if self._update_callback is not None and (epoch + 1) % log_interval == 0:
                try:
                    self._update_callback(record)
                except Exception:
                    pass

            if verbose and (epoch + 1) % max(log_interval * 5, 100) == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {loss_val:.3e} | "
                    f"PDE: {loss_f_val:.3e}"
                )

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

    def predict_grid(self, nx=100, nt=100, mc_samples=30):
        x_min, x_max = self.domain['x']
        t_min, t_max = self.domain['t']
        x = np.linspace(x_min, x_max, nx).astype(np.float32)
        t = np.linspace(t_min, t_max, nt).astype(np.float32)
        X_mesh, T_mesh = np.meshgrid(x, t)
        X_flat = np.hstack([X_mesh.reshape(-1, 1), T_mesh.reshape(-1, 1)]).astype(np.float32)

        if mc_samples > 1 and self.use_mc_dropout:
            U_mean_flat, U_std_flat = self.predict(X_flat, mc_samples=mc_samples)
            U_mean = U_mean_flat.reshape(nt, nx)
            U_std = U_std_flat.reshape(nt, nx)
            U_uncertainty = 2.0 * U_std
            return x, t, U_mean, U_std, U_uncertainty
        else:
            U_flat = self.predict(X_flat, mc_samples=1)
            U = U_flat.reshape(nt, nx)
            return x, t, U

    def predict_with_uncertainty(self, nx=100, nt=100, mc_samples=30):
        return self.predict_grid(nx, nt, mc_samples)

    def get_weights(self):
        weights = self.model.get_weights()
        serializable = []
        for w in weights:
            serializable.append({
                'data': w.tolist(),
                'shape': list(w.shape)
            })
        return serializable

    def set_weights(self, weights_data):
        weights = []
        for w in weights_data:
            weights.append(np.array(w['data']).reshape(w['shape']))
        self.model.set_weights(weights)
        self.is_trained = True


class PINNSolver(HardConstraintPINN):
    pass


def create_ic_func(expression_str, params=None):
    params = params or {}

    def ic_func(x):
        safe_dict = {
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
            'pi': np.pi, 'abs': np.abs, 'sqrt': np.sqrt,
            'tanh': np.tanh, 'sinh': np.sinh, 'cosh': np.cosh,
            'log': np.log, 'x': x, **{k: np.float32(v) for k, v in params.items()}
        }
        result = eval(expression_str, {"__builtins__": {}}, safe_dict)
        result = np.asarray(result, dtype=np.float32)
        if result.ndim == 0:
            result = np.full(x.shape if hasattr(x, 'shape') else (1,), result, dtype=np.float32)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        return result

    return ic_func


def create_bc_func(expression_str, params=None):
    params = params or {}

    def bc_func(t):
        safe_dict = {
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
            'pi': np.pi, 'abs': np.abs, 'sqrt': np.sqrt,
            'tanh': np.tanh, 'sinh': np.sinh, 'cosh': np.cosh,
            'log': np.log, 't': t, **{k: np.float32(v) for k, v in params.items()}
        }
        result = eval(expression_str, {"__builtins__": {}}, safe_dict)
        result = np.asarray(result, dtype=np.float32)
        if result.ndim == 0:
            result = np.full(t.shape if hasattr(t, 'shape') else (1,), result, dtype=np.float32)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        return result

    return bc_func
