from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        LSE_val = LogSumExp((1)).compute(Z)
        re_LSE_val = array_api.reshape(LSE_val, (LSE_val.shape[0], 1))
        return Z - re_LSE_val
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z, = node.inputs
        
        # First, get the softmax values
        # Note: node contains Z - LogSumExp(Z), which is LogSoftmax(Z)
        softmax_z = exp(node)
        out_grad_sum = summation(out_grad, axes=(1,))
        out_grad_sum = reshape(out_grad_sum, (out_grad_sum.shape[0], 1))
        out_grad_sum = broadcast_to(out_grad_sum, Z.shape)
        softmax_term = softmax_z*out_grad_sum
        return out_grad - softmax_term
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        """
        Compute logsumexp using NDArray methods
        """
        z_shape = Z.shape
        
        # For None axes case, we reduce over all dimensions
        if self.axes is None:
            max_z = Z.max(None)  # This gives us a scalar
            # We need to reshape to match input dimensions for broadcasting
            max_z = max_z.reshape((1,) * len(z_shape))
            max_z_expanded = max_z.broadcast_to(z_shape)
        else:
            # Get max along specified axes using NDArray's max method
            max_z = Z.max(self.axes)
            # Create a shape for broadcasting by inserting 1s at the reduced axes
            new_shape = list(max_z.shape)
            axes = (self.axes,) if isinstance(self.axes, int) else self.axes
            for axis in sorted(axes):
                new_shape.insert(axis, 1)
            # Reshape and broadcast
            max_z = max_z.reshape(tuple(new_shape))
            max_z_expanded = max_z.broadcast_to(z_shape)
            
        # Compute exp(Z - max_Z) and sum
        shift_z = Z - max_z_expanded
        exp_z = array_api.exp(shift_z)
        sum_z = exp_z.sum(self.axes)
        
        # For None axes case, sum_z and max_z will be scalars
        if self.axes is None:
            log_z = array_api.log(sum_z).reshape((1,))
            max_z = max_z.reshape((1,))
        else:
            log_z = array_api.log(sum_z)
            max_z = max_z.reshape(log_z.shape)
            
        return log_z + max_z

    def gradient(self, out_grad, node):
        Z, = node.inputs
        
        # Compute sum of exp(Z) along specified axes
        sum_exp_z = node
        se_z_shape = list(sum_exp_z.shape)
        
        # Create broadcasted shapes
        if self.axes is not None:
            axes = (self.axes,) if isinstance(self.axes, int) else self.axes
            for axis in sorted(axes):
                se_z_shape.insert(axis, 1)
        else:
            # For None axes case
            se_z_shape = [1] * len(Z.shape)

        # Reshape and broadcast sum_exp_z
        sum_exp_z = reshape(sum_exp_z, tuple(se_z_shape))
        sum_exp_z_broadcast = broadcast_to(sum_exp_z, Z.shape)

        # Reshape and broadcast out_grad
        out_grad_shape = list(out_grad.shape)
        if self.axes is not None:
            for axis in sorted(axes):
                out_grad_shape.insert(axis, 1)
        else:
            out_grad_shape = [1] * len(Z.shape)
            
        out_grad = reshape(out_grad, tuple(out_grad_shape))
        out_grad_broadcast = broadcast_to(out_grad, Z.shape)

        return out_grad_broadcast * exp(Z - sum_exp_z_broadcast)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)