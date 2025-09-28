import numpy as np
from collections import deque
from .base import Optimizer

class LBFGS(Optimizer):
    """
    Implements the L-BFGS algorithm.
    """
    def __init__(self, lr=1.0, history_size=10, max_iter=10, line_search_fn="backtracking"):
        super().__init__(lr)
        self.history_size = history_size
        self.max_iter = max_iter
        self.line_search_fn = line_search_fn
        self.s_history = deque(maxlen=self.history_size)
        self.y_history = deque(maxlen=self.history_size)
        self.prev_params = None
        self.prev_grads = None

    def _unflatten_params(self, flat_params, shapes):
        unflattened = []
        offset = 0
        for shape in shapes:
            num_elements = np.prod(shape)
            unflattened.append(flat_params[offset:offset + num_elements].reshape(shape))
            offset += num_elements
        return unflattened

    def _flatten_params(self, params):
        return np.concatenate([p.flatten() for p in params])

    def step(self, params, closure):
        """
        Performs a single optimization step.
        """
        param_shapes = [p.shape for p in params]
        current_params_flat = self._flatten_params(params)

        def eval_loss_and_grad(flat_params, backward=True):
            # Unflatten params for the model
            unflattened_params = self._unflatten_params(flat_params, param_shapes)

            # Update model params
            for i in range(len(params)):
                params[i][...] = unflattened_params[i]

            loss, grads = closure(backward=backward)
            flat_grads = None
            if grads is not None:
                flat_grads = self._flatten_params(grads)
            return loss, flat_grads

        # Initial evaluation
        current_loss, current_grads_flat = eval_loss_and_grad(current_params_flat, backward=True)

        if self.prev_params is not None:
            s = current_params_flat - self.prev_params
            y = current_grads_flat - self.prev_grads
            if np.dot(y, s) > 1e-10: # Curvature condition
                self.s_history.append(s)
                self.y_history.append(y)

        # Two-loop recursion to compute search direction
        q = current_grads_flat.copy()
        alphas = []
        for s, y in zip(reversed(self.s_history), reversed(self.y_history)):
            rho = 1.0 / np.dot(y, s)
            alpha = rho * np.dot(s, q)
            q -= alpha * y
            alphas.append(alpha)

        if len(self.s_history) > 0:
            s_latest = self.s_history[-1]
            y_latest = self.y_history[-1]
            gamma = np.dot(s_latest, y_latest) / np.dot(y_latest, y_latest)
            z = gamma * q
        else:
            z = q

        for (s, y), alpha in zip(zip(self.s_history, self.y_history), reversed(alphas)):
            rho = 1.0 / np.dot(y, s)
            beta = rho * np.dot(y, z)
            z += s * (alpha - beta)

        direction = -z

        # Backtracking line search
        alpha = self.lr
        c = 0.5
        tau = 0.5

        for _ in range(self.max_iter):
            new_params_flat = current_params_flat + alpha * direction
            new_loss, _ = eval_loss_and_grad(new_params_flat, backward=False)

            if new_loss <= current_loss + c * alpha * np.dot(current_grads_flat, direction):
                break
            alpha *= tau
        else: # if loop finishes without break
            new_params_flat = current_params_flat # No improvement found, stick to original params
            new_loss = current_loss

        # Update params
        final_params = self._unflatten_params(new_params_flat, param_shapes)
        for i in range(len(params)):
            params[i][...] = final_params[i]

        self.prev_params = current_params_flat.copy()
        self.prev_grads = current_grads_flat.copy()

        return new_loss