import numpy as np
from .base import Optimizer

class SOAP(Optimizer):
    """
    Implements SOAP algorithm.
    """
    def __init__(self, lr=0.003, betas=(0.95, 0.95), eps=1e-8, precondition_frequency=10, shampoo_beta=-1, correct_bias=True):
        super().__init__(lr)
        self.betas = betas
        self.shampoo_beta = shampoo_beta if shampoo_beta >= 0 else betas[1]
        self.eps = eps
        self.precondition_frequency = precondition_frequency
        self.correct_bias = correct_bias
        self.state = {}

    def step(self, params, closure):
        loss, grads = closure()

        for i, (p, grad) in enumerate(zip(params, grads)):
            if i not in self.state:
                self.state[i] = {}
                state = self.state[i]
                state["step"] = 0
                state["exp_avg"] = np.zeros_like(grad)
                state["exp_avg_sq"] = np.zeros_like(grad)
                self.init_preconditioner(grad, state)
                self.update_preconditioner(grad, state)
                continue

            state = self.state[i]

            grad_projected = self.project(grad, state)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = self.betas

            state["step"] += 1

            exp_avg *= beta1
            exp_avg += (1.0 - beta1) * grad_projected

            exp_avg_sq *= beta2
            exp_avg_sq += (1.0 - beta2) * np.square(grad_projected)

            denom = np.sqrt(exp_avg_sq) + self.eps

            step_size = self.lr
            if self.correct_bias:
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

            norm_grad = self.project_back(exp_avg / denom, state)

            p -= step_size * norm_grad

            self.update_preconditioner(grad, state)

        return loss

    def init_preconditioner(self, grad, state):
        state['GG'] = []
        if grad.ndim == 1:
            state['GG'].append(np.zeros((grad.shape[0], grad.shape[0])))
        else:
            for sh in grad.shape:
                state['GG'].append(np.zeros((sh, sh)))

        state['Q'] = None
        state['shampoo_beta'] = self.shampoo_beta

    def project(self, grad, state):
        if state['Q'] is None:
            return grad

        projected_grad = grad
        for mat in state['Q']:
            if len(mat) > 0:
                projected_grad = np.tensordot(projected_grad, mat, axes=([0], [0]))
            else:
                permute_order = list(range(1, len(projected_grad.shape))) + [0]
                projected_grad = projected_grad.transpose(permute_order)
        return projected_grad

    def project_back(self, grad, state):
        if state['Q'] is None:
            return grad

        projected_grad = grad
        for mat in state['Q']:
            if len(mat) > 0:
                projected_grad = np.tensordot(projected_grad, mat, axes=([0], [1]))
            else:
                permute_order = list(range(1, len(projected_grad.shape))) + [0]
                projected_grad = projected_grad.transpose(permute_order)
        return projected_grad

    def update_preconditioner(self, grad, state):
        if state["Q"] is not None:
             state["exp_avg"] = self.project_back(state["exp_avg"], state)

        if grad.ndim == 1:
            outer_product = np.outer(grad, grad)
            state['GG'][0] = (1 - state['shampoo_beta']) * outer_product + state['shampoo_beta'] * state['GG'][0]

        else:
            for idx, sh in enumerate(grad.shape):
                dims = list(range(grad.ndim))
                del dims[idx]
                outer_product = np.tensordot(grad, grad, axes=(dims, dims))
                state['GG'][idx] = (1-state['shampoo_beta']) * outer_product + state['shampoo_beta'] * state['GG'][idx]

        if state['Q'] is None:
            state['Q'] = self.get_orthogonal_matrix(state['GG'])

        if state['step'] > 0 and state['step'] % self.precondition_frequency == 0:
            state['Q'] = self.get_orthogonal_matrix(state['GG'])

        if state["step"] > 0:
            state["exp_avg"] = self.project(state["exp_avg"], state)

    def get_orthogonal_matrix(self, mat_list):
        matrix = []
        for m in mat_list:
            if len(m) == 0:
                matrix.append([])
                continue

            try:
                _, Q = np.linalg.eigh(m + 1e-30 * np.eye(m.shape[0]))
            except np.linalg.LinAlgError:
                _, Q = np.linalg.eigh(m.astype(np.float64) + 1e-30 * np.eye(m.shape[0]))
                Q = Q.astype(m.dtype)

            Q = np.flip(Q, axis=1)
            matrix.append(Q)
        return matrix