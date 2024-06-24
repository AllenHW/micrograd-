import numpy as np

from base import Op, UnaryOp
from ..tensor.base import Tensor

class Tranpose(Op):
  OP = 'transpose'

  def __init__(self, a, axes=None):
    self.a = a
    self.axes = axes
    self.out = None

  def forward(self):
    self.out = Tensor(np.transpose(self.a, self.axes), (self.a,), self)
    return self.out
  
  def backward(self):
    if self.axes is None:
      self.a.grad += np.tranpose(self.out.grad)
    else:
      inv = [None] * len(self.axes)
      for i,idx in self.axes:
        inv[idx] = i
      self.a.grad += np.transpose(self.out.grad.transpose, axes=inv)

class Swapaxes(Op):
  OP = 'swapaxes'

  def __init__(self, a, axis1, axis2):
    self.a = a
    self.axis1 = axis1
    self.axis2 = axis2
    self.out = None

  def forward(self):
    self.out = Tensor(np.swapaxes(self.a, self.axis1, self.axis2), (self.a,), self)
    return self.out
  
  def backward(self):
    self.a.grad += np.swapaxes(self.out.grad, self.axis1, self.axis2)


class Ravel(UnaryOp):
  OP = 'ravel'

  def forward(self):
    self.out = Tensor(np.ravel(self.a), (self.a,), self)
    return self.out
  
  def backward(self):
    self.a.grad += np.reshape(self.out.grad, self.grad.shape)


class Reshape(Op):
  OP = 'reshape'

  def __init__(self, a, new_shape):
    self.a = a
    self.new_shape = new_shape
    self.out = None
  
  def forward(self):
    self.out = Tensor(np.reshape(self.a, self.new_shape), (self.a,), self)
    return self.out
  
  def backward(self):
    self.a.grad += np.reshape(self.out.grad, self.grad.shape)


class Squeeze(Op):
  OP = 'squeeze'

  def __init__(self, a, axis=None):
    self.a = a
    self.axis = axis
    self.out = None
  
  def forward(self):
    self.out = Tensor(np.squeeze(self.a, self.axis), (self.a,), self)
    return self.out
  
  def backward(self):
    self.a.grad += np.expand_dims(self.out.grad, self.axis)


class Expanddims(Op):
  OP = 'expanddims'

  def __init__(self, a, axis):
    self.a = a
    self.axis = axis
    self.out = None
  
  def forward(self):
    self.out = Tensor(np.expand_dims(self.a, self.axis), (self.a,), self)
    return self.out
  
  def backward(self):
    self.a.grad += np.squeeze(self.out.grad, self.axis)


class Concat(Op):
  OP = 'concat'

  def __init__(self, xs, axis=0):
    self.xs = xs 
    self.axis = axis
    self.out = None

  def forward(self):
    self.out = Tensor(np.concat(self.xs, self.axis), self.xs, self)
    return self.out
  
  def backward(self):
    idx = 0
    for x in self.xs:
      size = x.shape[self.axis]
      x.grad += np.take(self.out.grad, range(idx, idx + size), axis=self.axis)
      idx += size
