"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.kaiming_uniform(in_features, out_features, None, "relu", requires_grad=True, device=device, dtype=dtype))
        if bias:
          # self.bias=Parameter(init.kaiming_uniform(out_features, 1, None, "relu", device=device, dtype=dtype))
          # self.bias=Parameter(ops.reshape(self.bias, (1, out_features)))
          self.bias = Parameter(ops.transpose(init.init_initializers.kaiming_uniform(fan_in=self.out_features,  
                                                  fan_out=1, 
                                                  requires_grad=True,
                                                  device=device,
                                                  dtype=dtype))) 
        else:
          self.bias=None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias:
          N = X.shape[0]
          return ops.matmul(X, self.weight) + ops.broadcast_to(self.bias, (N, self.out_features))
        else:
          return ops.matmul(X, self.weight)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        res = 1
        for i, s in enumerate(X.shape):
          if i != 0:
            res *= s
        return ops.reshape(X, (X.shape[0], res))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        z_onehot = init.one_hot(logits.shape[1], y, device=logits.device)
        LSE = ops.LogSumExp((1,))(logits)
        # print("softmax", z_onehot.device, y.device, logits.device)
        z_y = ops.summation(logits*z_onehot, (1,))
        return ops.summation(LSE - z_y) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))  # gamma (scale)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))   # beta (shift)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ### TODO: decide training?
        if self.training:
            # Compute the batch mean
            batch_mean = ops.summation(x, axes=(0,)) / x.shape[0]
            batch_mean_reshaped = ops.reshape(batch_mean, (1, x.shape[1]))  # Reshape to (1, dim)
            batch_mean_broadcasted = ops.broadcast_to(batch_mean_reshaped, x.shape)  # Broadcast to match (batch_size, dim)

            # Compute the batch variance
            batch_var = ops.summation(ops.power_scalar(x - batch_mean_broadcasted, 2), axes=(0,)) / x.shape[0]
            batch_var_reshaped = ops.reshape(batch_var, (1, x.shape[1]))  # Reshape to (1, dim)
            batch_var_broadcasted = ops.broadcast_to(batch_var_reshaped, x.shape)  # Broadcast to match (batch_size, dim)
            
            # Detach the batch mean and variance to avoid tracking gradients
            batch_mean_detached = batch_mean.detach()
            batch_var_detached = batch_var.detach()
            
            # Update running statistics using momentum (detached versions)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean_detached
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var_detached
            
            # Normalize the batch
            normalized_x = (x - batch_mean_broadcasted) / ops.power_scalar(batch_var_broadcasted + self.eps, 0.5)
        else:
            # During inference, use running mean and variance
            running_mean_broadcasted = ops.broadcast_to(ops.reshape(self.running_mean, (1, x.shape[1])), x.shape)
            running_var_broadcasted = ops.broadcast_to(ops.reshape(self.running_var, (1, x.shape[1])), x.shape)
            
            # Normalize using running statistics
            normalized_x = (x - running_mean_broadcasted) / ops.power_scalar(running_var_broadcasted + self.eps, 0.5)
        
        # Apply learnable parameters (scale and shift)
        weight_broadcasted = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        bias_broadcasted = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)

        return normalized_x * weight_broadcasted + bias_broadcasted
        ### END YOUR SOLUTION

# class BatchNorm2d(BatchNorm1d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, x: Tensor):
#         # nchw -> nhcw -> nhwc
#         s = x.shape
#         _x = x.transpose((1, 2)).transpose((2, 3))
#         _x_compact = Tensor(_x.cached_data.compact(), device=_x.device)  # Make compact
#         _x_reshaped = _x_compact.reshape((s[0] * s[2] * s[3], s[1]))  # Reshape to 2D
#         y = super().forward(_x_reshaped).reshape((s[0], s[2], s[3], s[1]))
#         return y.transpose((2,3)).transpose((1,2))

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.init_initializers.ones(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.init_initializers.zeros(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes=(1,)) / x.shape[1]
        mean = ops.reshape(mean, (x.shape[0], 1))
        mean = ops.broadcast_to(mean, x.shape)
        xm_diff = x-mean
        
        variance = ops.summation(ops.power_scalar(xm_diff, 2), axes=(1,)) / x.shape[1]
        variance = ops.reshape(variance, (x.shape[0], 1))
        variance = ops.broadcast_to(variance, x.shape)

        normalized_x = (x - mean) / ops.power_scalar(variance + self.eps, 0.5)


        weight_broadcast = ops.broadcast_to(self.weight, x.shape)
        bias_broadcast = ops.broadcast_to(self.bias, x.shape)

        return normalized_x * weight_broadcast + bias_broadcast

        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### TODO: decide training?
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype)
            return x * mask / (1 - self.p)
        else:
            # During evaluation, dropout does not modify the input
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION