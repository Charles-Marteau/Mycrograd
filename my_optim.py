from my_engine import Tnsr
import numpy as np
import my_nn as nn


# Definition of an optimizer which interacts with my_nn.py
# and my_engine.py (i.e. with our layers and our Tnsr class)
# The two available optimizers are 'stochastic gradient descent'
# and 'ADAM'.

# We follow the same parametrization as the one in pytorch.


class optimizer():
    """
    Methods and attributes inherited by all the optimizers
    """
    def __init__(self, params, lr=1e-03):
        self.params = params
        self.lr = lr
        
    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data)

class SGD(optimizer):

    def __init__(self, params, lr):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data += - self.lr * param.grad

class Adam(optimizer):

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params, lr)
        self.betas = betas
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = [np.zeros_like(param.data) for param in params] # Initialize the 1st moment
        self.v = [np.zeros_like(param.data) for param in params] # Initialize the 2nd moment
        self.steps = 1

    def step(self):
        for k, param in enumerate(self.params):
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * param.grad
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (param.grad)**2
            m_hat = self.m[k].copy() / (1 - self.beta1**self.steps)
            v_hat = self.v[k].copy() / (1 - self.beta2**self.steps)
            param.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)
            self.steps += 1

class clip_grad_norm():

    def __init__(self, params, max_abs):
        for param in params:
            mask = np.abs(param.grad) >= max_abs
            param.grad[mask] == max_abs


     
            





    

