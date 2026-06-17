import numpy as np
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


class PINNSolver:
    def __init__(self, pde_type='heat', domain=None, layers=None, params=None):
        self.pde_type = pde_type
        self.domain = domain or {'x': (0.0, 1.0), 't': (0.0, 1.0)}
        self.layers = layers or [2, 64, 64, 64, 1]
        self.params = params or {'alpha': 0.01}
        self.model = self._build_network()
        self.history = []
        self.is_trained = False

    def _build_network(self):
        inputs = tf.keras.Input(shape=(self.layers[0],))
        x = inputs
        for width in self.layers[1:-1]:
            x = tf.keras.layers.Dense(
                width, activation='tanh',
                kernel_initializer='glorot_normal'
            )(x)
        outputs = tf.keras.layers.Dense(
            self.layers[-1], activation=None,
            kernel_initializer='glorot_normal'
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
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
                u = self.model(tf.concat([x, t], axis=1))
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
            residual = u_tt - c**2 * u_xx
        elif self.pde_type == 'elliptic':
            residual = -alpha * u_xx
        else:
            residual = u_t - alpha * u_xx
        return residual

    def _generate_collocation_points(self, N_f):
        x_min, x_max = self.domain['x']
        t_min, t_max = self.domain['t']
        X_f = np.random.uniform(
            x_min, x_max, (N_f, 1)
        )
        T_f = np.random.uniform(
            t_min, t_max, (N_f, 1)
        )
        return np.hstack([X_f, T_f])

    def _generate_initial_points(self, N_i, ic_func=None):
        x_min, x_max = self.domain['x']
        X_i = np.linspace(x_min, x_max, N_i).reshape(-1, 1)
        T_i = np.zeros((N_i, 1))
        if ic_func is not None:
            U_i = ic_func(X_i)
        else:
            U_i = np.sin(np.pi * X_i)
        return np.hstack([X_i, T_i]), U_i

    def _generate_boundary_points(self, N_b, bc_funcs=None):
        x_min, x_max = self.domain['x']
        t_min, t_max = self.domain['t']
        T_b = np.linspace(t_min, t_max, N_b).reshape(-1, 1)
        X_left = np.full((N_b, 1), x_min)
        X_right = np.full((N_b, 1), x_max)
        X_bnd = np.vstack([X_left, X_right])
        T_bnd = np.vstack([T_b, T_b])
        if bc_funcs is not None:
            U_left = bc_funcs[0](T_b)
            U_right = bc_funcs[1](T_b)
            U_bnd = np.vstack([U_left, U_right])
        else:
            U_bnd = np.zeros((2 * N_b, 1))
        return np.hstack([X_bnd, T_bnd]), U_bnd

    def train(self, ic_func=None, bc_funcs=None,
              N_f=10000, N_i=200, N_b=200,
              epochs=5000, learning_rate=1e-3, verbose=True):
        X_f = self._generate_collocation_points(N_f)
        X_f = tf.constant(X_f, dtype=tf.float32)

        X_i, U_i = self._generate_initial_points(N_i, ic_func)
        X_i = tf.constant(X_i, dtype=tf.float32)
        U_i = tf.constant(U_i, dtype=tf.float32)

        X_b, U_b = self._generate_boundary_points(N_b, bc_funcs)
        X_b = tf.constant(X_b, dtype=tf.float32)
        U_b = tf.constant(U_b, dtype=tf.float32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.history = []

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                u_pred_i = self.model(X_i)
                loss_i = tf.reduce_mean(tf.square(u_pred_i - U_i))

                u_pred_b = self.model(X_b)
                loss_b = tf.reduce_mean(tf.square(u_pred_b - U_b))

                f_pred = self._pde_residual(X_f)
                loss_f = tf.reduce_mean(tf.square(f_pred))

                loss = loss_i + loss_b + loss_f

            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            loss_val = loss.numpy().item()
            self.history.append({
                'epoch': epoch + 1,
                'loss': loss_val,
                'loss_ic': loss_i.numpy().item(),
                'loss_bc': loss_b.numpy().item(),
                'loss_pde': loss_f.numpy().item(),
            })

            if verbose and (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_val:.6e}")

        self.is_trained = True
        return self.history

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        return self.model(X).numpy()

    def predict_grid(self, nx=100, nt=100):
        x_min, x_max = self.domain['x']
        t_min, t_max = self.domain['t']
        x = np.linspace(x_min, x_max, nx)
        t = np.linspace(t_min, t_max, nt)
        X, T = np.meshgrid(x, t)
        X_flat = np.hstack([X.reshape(-1, 1), T.reshape(-1, 1)])
        U_flat = self.predict(X_flat)
        U = U_flat.reshape(nt, nx)
        return x, t, U

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


def create_ic_func(expression_str, params=None):
    params = params or {}
    def ic_func(x):
        safe_dict = {
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
            'pi': np.pi, 'abs': np.abs, 'sqrt': np.sqrt,
            'x': x, **{k: np.float32(v) for k, v in params.items()}
        }
        result = eval(expression_str, {"__builtins__": {}}, safe_dict)
        result = np.array(result, dtype=np.float32)
        if result.ndim == 0:
            result = np.full_like(x, result, dtype=np.float32)
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
            't': t, **{k: np.float32(v) for k, v in params.items()}
        }
        result = eval(expression_str, {"__builtins__": {}}, safe_dict)
        result = np.array(result, dtype=np.float32)
        if result.ndim == 0:
            result = np.full_like(t, result, dtype=np.float32)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        return result
    return bc_func
