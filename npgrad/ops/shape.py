import numpy as np

from .base import Op, UnaryOp

class Transpose(Op):
  OP = 'transpose'

  def __init__(self, x, axes=None):
    super().__init__()
    self.x = x
    self.axes = axes
    self.out = None

    self.ndarray_cls = self.x.__class__

  def forward(self):
    data = np.transpose(self.x.data, self.axes)
    self.out = self.ndarray_cls(data, (self.x,), self)
    return self.out
    
  def backward(self):
    if self.axes is None:
      self.x.grad += np.transpose(self.out.grad)
    else:
      inv_axes = np.argsort(self.axes)
      self.x.grad += np.transpose(self.out.grad, axes=inv_axes)


class Swapaxes(Op):
  OP = 'swapaxes'

  def __init__(self, x, axis1, axis2):
    self.x = x
    self.axis1 = axis1
    self.axis2 = axis2
    self.out = None

    self.ndarray_cls = self.x.__class__

  def forward(self):
    data = np.swapaxes(self.x.data, self.axis1, self.axis2)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out
  
  def backward(self):
    self.x.grad += np.swapaxes(self.out.grad, self.axis1, self.axis2)


class Ravel(UnaryOp):
  OP = 'ravel'

  def forward(self):
    data = np.ravel(self.x.data)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out
  
  def backward(self):
    self.x.grad += np.reshape(self.out.grad, self.x.grad.shape)


class Reshape(Op):
  OP = 'reshape'

  def __init__(self, x, newshape):
    self.x = x
    self.newshape = newshape
    self.out = None

    self.ndarray_cls = self.x.__class__
  
  def forward(self):
    data = np.reshape(self.x.data, self.newshape)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out
  
  def backward(self):
    self.x.grad += np.reshape(self.out.grad, self.x.grad.shape)


class Squeeze(Op):
  OP = 'squeeze'

  def __init__(self, x, axis=None):
    self.x = x
    self.axis = axis
    self.out = None

    self.ndarray_cls = self.x.__class__
  
  def forward(self):
    data = np.squeeze(self.x.data, self.axis)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out
  
  def backward(self):
    self.x.grad += np.reshape(self.out.grad, self.x.grad.shape)


class Expanddims(Op):
  OP = 'expanddims'

  def __init__(self, x, axis):
    self.x = x
    self.axis = axis
    self.out = None

    self.ndarray_cls = self.x.__class__
  
  def forward(self):
    data = np.expand_dims(self.x.data, self.axis)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out
  
  def backward(self):
    self.x.grad += np.squeeze(self.out.grad, self.axis)


class Concat(Op):
  OP = 'concat'

  def __init__(self, xs, axis=0):
    self.xs = xs 
    self.axis = axis
    self.out = None

    self.ndarray_cls = self.xs[0].__class__

  def forward(self):
    data = np.concatenate([x.data for x in self.xs], self.axis)
    self.out = self.ndarray_cls(data, self.xs, self)

    return self.out
  
  def backward(self):
    idx = 0
    for x in self.xs:
      size = x.shape[self.axis]
      x.grad += np.take(self.out.grad, range(idx, idx + size), axis=self.axis)
      idx += size
