from .ops.index import MaskedFill
from .ops.repeats import Repeat, Tile
from .ops.reduction import Sum, Mean, Max, Min
from .ops.shape import Transpose, Reshape, Swapaxes, Ravel, Squeeze, Expanddims, Concat
from .ops.unary import Log, Exp, Sin, Cos, Tan, Relu, Tanh 
from .ops.tensors import Matmul, Einsum 
from .ops.pooling import MaxPool2D, AvgPool2D, Conv2D

from .ndarray import ndarray


### Index Operations

def masked_fill(x, mask, value):
  op = MaskedFill(x, mask, value)
  return op.forward()

### Repeat Operations

def repeat(x, repeats, axis=None):
  op = Repeat(x, repeats, axis)
  return op.forward()

def tile(x, reps):
  op = Tile(x, reps)
  return op.forward()


### Reduction Operations

def sum(x, axis=None, keepdims=False):
  op = Sum(x, axis=axis, keepdims=keepdims)
  return op.forward()

def mean(x, axis=None, keepdims=False):
  op = Mean(x, axis=axis, keepdims=keepdims)
  return op.forward()

def max(x, axis=None, keepdims=False):
  op = Max(x, axis=axis, keepdims=keepdims)
  return op.forward()

def min(x, axis=None, keepdims=False):
  op = Min(x, axis=axis, keepdims=keepdims)
  return op.forward()


### Shape Operations

def transpose(x, axes=None):
  op = Transpose(x, axes)
  return op.forward()

def swapaxes(x, axis1, axis2):
  op = Swapaxes(x, axis1, axis2)
  return op.forward()

def ravel(x):
  op = Ravel(x)
  return op.forward()

def reshape(x, newshape):
  op = Reshape(x, newshape)
  return op.forward()

def squeeze(x, axis=None):
  op = Squeeze(x, axis)
  return op.forward()

def expand_dims(x, axis):
  op = Expanddims(x, axis)
  return op.forward()

def concat(xs, axis=0):
  op = Concat(xs, axis)
  return op.forward()

def concatenate(xs, axis=0):
  return concat(xs, axis)


### Unary Operations

def log(x):
  op = Log(x)
  return op.forward()

def exp(x):
  op = Exp(x)
  return op.forward()

def sin(x):
  op = Sin(x)
  return op.forward()

def cos(x):
  op = Cos(x)
  return op.forward()

def tan(x):
  op = Tan(x)
  return op.forward()

def relu(x):
  op = Relu(x)
  return op.forward()

def tanh(x):
  op = Tanh(x)
  return op.forward()


### Tensor Operations

def matmul(a, b):
  op = Matmul(a, b)
  return op.forward()

def einsum(subscripts, *operands):
  op = Einsum(subscripts, *operands)
  return op.forward()


### Pooling Operations

def max_pool2d(x, kernel_size, stride=1, padding=1, dilation=1):
  op = MaxPool2D(x, kernel_size, stride, padding, dilation)
  return op.forward()

def avg_pool2d(x, kernel_size, stride=1, padding=1, dilation=1):
  op = AvgPool2D(x, kernel_size, stride, padding, dilation)
  return op.forward()

def conv2d(x, weight, bias, stride=1, padding=1, dilation=1):
  op = Conv2D(x, weight, bias, stride, padding, dilation)
  return op.forward()


### Constructions

def zeros(shape):
  return ndarray.zeros(shape)

def ones(shape):
  return ndarray.ones(shape)

def randn(*shape):
  return ndarray.randn(*shape)

def eye(n, m=None):
  return ndarray.eye(n, m)

def arange(start, stop=None, step=1):
  return ndarray.arange(start, stop, step)


def array(data):
  return ndarray(data)
