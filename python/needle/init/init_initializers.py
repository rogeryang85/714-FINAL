import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain*math.sqrt(6/(fan_in+fan_out))
    res = rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION
    return res


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain*math.sqrt(2/(fan_in+fan_out))
    res = randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION
    return res



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    
    if shape is not None:
        # Calculate fan_in and fan_out based on shape for convolution
        print("kaiming_uniform", shape)
        receptive_field_size = shape[0] * shape[1] if len(shape) > 2 else 1
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        # Bound for uniform distribution
        bound = gain * math.sqrt(3 / fan_in)
        res = rand(*shape, low=-bound, high=bound, **kwargs)
    else:
        # Use provided fan_in and fan_out for fully-connected layers
        bound = gain * math.sqrt(3 / fan_in)
        res = rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    
    return res
    
def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain/math.sqrt(fan_in)
    res = randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION
    return res