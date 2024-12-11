"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return array_api.power(a, b)
        
    def gradient(self, out_grad, node):
        x, y = node.inputs
        # print(type(x))
        return out_grad*y*power(x, y-1), out_grad*power(x, y)*log(x)


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        (x,) = node.inputs
        return (out_grad*self.scalar*power_scalar(x, self.scalar-1),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a/b

    def gradient(self, out_grad, node):
      (x, y) = node.inputs
      return (out_grad/y, -out_grad*x*power_scalar(y, -2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a/self.scalar

    def gradient(self, out_grad, node):
        return (out_grad/self.scalar, )

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ndim = len(a.shape)
        dims = list(range(ndim))
        
        if self.axes:
            # Only swap axes that are within bounds
            if max(self.axes) < ndim:
                dims[self.axes[0]], dims[self.axes[1]] = dims[self.axes[1]], dims[self.axes[0]]
        else:
            # Default behavior - swap last two dimensions
            if ndim > 1:
                dims[-1], dims[-2] = dims[-2], dims[-1]
                
        return array_api.transpose(a, dims)

    def gradient(self, out_grad, node):
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a.compact(), self.shape)

    def gradient(self, out_grad, node):
        x, = node.inputs
        x_shape = x.shape
        res = reshape(out_grad, x_shape)
        return res


def reshape(a, shape):
    return Reshape(shape)(a)

def zip_longest(iter1, iter2, fillvalue=None):
        iter1, iter2 = list(iter1), list(iter2)
        max_len = max(len(iter1), len(iter2))
        for i in range(max_len):
            val1 = iter1[i] if i < len(iter1) else fillvalue
            val2 = iter2[i] if i < len(iter2) else fillvalue
            yield (val1, val2)

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        x, = node.inputs
        input_shape = x.shape
        # Create a list to store the axes that were broadcasted
        broadcasted_axes = []
        input_shape_reversed = list(reversed(input_shape))
        grad_shape_reversed = list(reversed(out_grad.shape))
        
        # Find the axes that were broadcasted
        for i, (input_dim, grad_dim) in enumerate(zip_longest(input_shape_reversed, 
                                                             grad_shape_reversed, 
                                                             fillvalue=1)):
            if input_dim != grad_dim:
                broadcasted_axes.append(len(grad_shape_reversed) - i - 1)
        
        # Sum out the broadcasted dimensions
        grad = out_grad
        broadcasted_axes.sort(reverse=True)
        for axis in broadcasted_axes:
            grad = summation(grad, axes=(axis))
        
        # Squeeze any extra dimensions that were added
        if len(input_shape) < len(self.shape):
            n_dims_added = len(self.shape) - len(input_shape)
            grad = reshape(grad, input_shape)
        
        return grad


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        if isinstance(self.axes, int):
            return array_api.sum(a, (self.axes,))
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        x, = node.inputs
        input_shape = x.shape
        
        if self.axes is None:
            out_grad = reshape(out_grad, (1,) * len(input_shape))
            return broadcast_to(out_grad, input_shape)
            
        axes = (self.axes,) if isinstance(self.axes, int) else self.axes
        
        output_shape = list(out_grad.shape)
        for axis in sorted(axes):
            output_shape.insert(axis, 1)
        out_grad = reshape(out_grad, tuple(output_shape))
        return broadcast_to(out_grad, input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # print("a", a.shape, "b", b.shape)
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        (x, y) = node.inputs
        x_shape = x.shape
        y_shape = y.shape
        trans_y = transpose(y)
        trans_x = transpose(x)
        x_grad = matmul(out_grad, trans_y)
        y_grad = matmul(trans_x, out_grad)
        if x_grad.shape != x_shape:
          # print(f"matmul_grad: x_grad: {x_grad.shape}; x: {x_shape}")
          batch_dims = tuple(range(len(x_grad.shape) - len(x_shape)))
          x_grad = summation(x_grad, axes=batch_dims)
        if y_grad.shape != y_shape:
          batch_dims = tuple(range(len(y_grad.shape) - len(y_shape)))
          y_grad = summation(y_grad, axes=batch_dims)
        return (x_grad, y_grad)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        x, = node.inputs
        return out_grad/x


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        x, = node.inputs
        return exp(x)*out_grad


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        self.x = (a > 0.0)
        return self.x * a

    def gradient(self, out_grad, node):
        x = Tensor(self.x, device=out_grad.device)
        # print(f"ReLU: out_grad: {out_grad.shape}, {out_grad.device}; x: {x.shape}, {x.device}; in: {node.inputs[0].shape}, {node.inputs[0].device}")
        return x * out_grad


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        t = tanh(x)
        # Create ones on the same device as the input
        ones = init_basic.ones(*t.shape, device=x.device)
        return out_grad * (ones - multiply(t, t))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)

class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        array_shape = args[0].shape
        n = len(args)
        
        # Calculate new shape by inserting n at the axis position
        new_shape = list(array_shape)
        new_shape.insert(self.axis, 1)
        
        # First reshape each array to have a size-1 dimension at axis
        arrays = []
        for arr in args:
            reshaped = arr.reshape(new_shape)
            arrays.append(reshaped)
        
        # Initialize the result array with the correct final shape
        final_shape = list(array_shape)
        final_shape.insert(self.axis, n)
        result = NDArray.make(tuple(final_shape), device=args[0].device)
        
        # Copy each array into its proper position
        for i, arr in enumerate(arrays):
            # Calculate the slice for this array
            slices = [slice(None) for _ in range(len(final_shape))]
            slices[self.axis] = slice(i, i+1)
            result.__setitem__(tuple(slices), arr)
        
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION



def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        A = A.compact()
        
        split_size = A.shape[self.axis]
        results = []
        
        for i in range(split_size):
            # Create slices for this section
            slices = [slice(None) for _ in range(len(A.shape))]
            slices[self.axis] = slice(i, i+1)
            # Get the section
            section = A.__getitem__(tuple(slices))
            
            # Make the section compact before reshaping
            section = section.compact()
            
            # Calculate the new shape by removing the axis dimension
            new_shape = list(section.shape)
            del new_shape[self.axis]
            
            # Reshape the compacted section
            section = section.reshape(tuple(new_shape))
            results.append(section)
            
        return tuple(results)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
    
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION
    
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Flipping again along the same axes brings us back to the original order
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Calculate new shape with dilation
        old_shape = a.shape
        new_shape = list(old_shape)
        
        # For each axis to dilate, increase its dimension
        # Don't try to access invalid axes
        valid_axes = [ax for ax in self.axes if ax < len(old_shape)]
        for axis in valid_axes:
            new_shape[axis] = old_shape[axis] * (self.dilation + 1)
            
        # Create output array filled with zeros
        out = array_api.full(new_shape, 0, dtype="float32", device=a.device)
        
        # Create slicing tuples to place original data
        slices = []
        for i in range(len(new_shape)):
            if i in valid_axes:
                slices.append(slice(0, None, self.dilation + 1))
            else:
                slices.append(slice(None))
                
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    # Handle both integer and tuple axes
    if isinstance(axes, int):
        axes = (axes,)
    
    # Keep only valid axes for the array's dimension
    ndim = a.ndim if hasattr(a, 'ndim') else len(a.shape)
    axes = tuple([ax if ax >= 0 else ndim + ax for ax in axes])

    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Only use valid axes
        valid_axes = [ax for ax in self.axes if ax < len(a.shape)]
        
        slices = []
        for i in range(len(a.shape)):
            if i in valid_axes:
                slices.append(slice(0, None, self.dilation + 1))
            else:
                slices.append(slice(None))
                
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    # Handle both integer and tuple axes
    if isinstance(axes, int):
        axes = (axes,)
    
    # Keep only valid axes for the array's dimension
    ndim = a.ndim if hasattr(a, 'ndim') else len(a.shape)
    axes = tuple([ax if ax >= 0 else ndim + ax for ax in axes])

    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        """
        Compute 2D convolution using NDArray operations
        A: NHWC format (batch_size, height, width, in_channels)
        B: KHWI format (kernel_size, kernel_size, in_channels, out_channels)
        """
        # Get dimensions
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        
        # Pad input if necessary
        if self.padding > 0:
            pad_width = ((0, 0), (self.padding, self.padding),
                        (self.padding, self.padding), (0, 0))
            A = A.pad(pad_width)
            H_padded = H + 2 * self.padding
            W_padded = W + 2 * self.padding
        else:
            H_padded, W_padded = H, W
        
        # print("N, H, W, C_in: ", (N, H, W, C_in))
        # print("K, C_out, self.padding, self.stride: ", (K, C_out, self.padding, self.stride))
        
        # Calculate output dimensions
        H_out = max(1, (H_padded - K) // self.stride + 1)
        W_out = max(1, (W_padded - K) // self.stride + 1)
        
        # Initialize output array
        # print("N, H_out, W_out, C_out: ", (N, H_out, W_out, C_out))
        out = NDArray.make((N, H_out, W_out, C_out), device=A.device)
        out.fill(0)
        
        # Make weight matrix compact and reshape once
        B_compact = B.compact()
        weight_reshaped = B_compact.reshape((K * K * C_in, C_out)) # Error happens before reshaping B
        # print("Before for loop")
        # Compute convolution
        for n in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * self.stride
                    w_start = w * self.stride
                    
                    # Extract input patch
                    patch = A.__getitem__((
                        slice(n, n+1),
                        slice(h_start, h_start + K),
                        slice(w_start, w_start + K),
                        slice(None)
                    ))
                    
                    # Make patch compact before reshaping
                    patch = patch.compact()
                    patch_reshaped = patch.reshape((1, K * K * C_in))
                    
                    # Compute output for this position
                    output = patch_reshaped.__matmul__(weight_reshaped)
                    out.__setitem__((n, h, w, slice(None)), output.reshape((C_out,)))
        
        return out.compact()

    def gradient(self, out_grad, node):
      """
      Compute gradients for Conv operation.
      out_grad: gradient with respect to output (N, H_out, W_out, C_out)
      X: input (N, H, W, C_in)
      W: weights (K, K, C_in, C_out)
      """
      X, W = node.inputs
      K = W.shape[0]  # Kernel size
      
      # Handle strided convolution by dilating out_grad
      if self.stride > 1:
          out_grad = dilate(out_grad, (1, 2), self.stride - 1)
      
      # X.grad computation
      # First, flip the weight kernel in both spatial dimensions
      W_flipped = flip(W, (0, 1))
      # Transpose weights from (K,K,C_in,C_out) to (K,K,C_out,C_in)
      W_flipped = transpose(W_flipped, (2, 3))
      
      # Calculate padding needed for X.grad
      if self.padding > 0:
          p = K - 1 - self.padding
      else:
          p = K - 1
      
      # Compute X.grad using convolution
      X_grad = conv(out_grad, W_flipped, stride=1, padding=p)
      
      # W.grad computation
      # For W.grad, we need X and out_grad to convolve
      # Reshape X from (N,H,W,C_in) to (C_in,H,W,N)
      X_reshaped = transpose(X, (0, 3))
      # Reshape out_grad from (N,H,W,C_out) to (C_out,H,W,N)
      out_grad_reshaped = transpose(out_grad, (0, 1))
      out_grad_reshaped = transpose(out_grad_reshaped, (1, 2))
      
      # Compute convolution for W.grad, Error happens in this step!!
      W_grad = conv(X_reshaped, out_grad_reshaped, 
                  stride=1, padding=self.padding)
      # print(f"W_grad:{W_grad.shape}")
      W_grad = transpose(W_grad, (0, 1)) # (3, 8, 3, 16)
      W_grad = transpose(W_grad, (1, 2)) # (3, 3, 8, 16)
      
      return X_grad, W_grad


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)