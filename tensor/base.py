import numpy as np

class Tensor:
  def __init__(self, data: np.ndarray, _prev=(), _op=None):
    self.data = data
    self._prev = set(_prev)
    self._op = _op
    self.grad = None

  def __repr__(self):
    return f"Tensor(data={self.data}, grad={self.grad})"
  
  def __add__(self, other):  # self + other
    return self._add_impl(other)

  def __mul__(self, other):  # self * other
    return self._mul_impl(other)
  
  def __pow__(self, other): # self ** other
    return self._pow_impl(other) 
  
  def __matmul__(self, other):
    return self._matmul_impl(other)

  def __getitem__(self, key):
    return self._getitem_impl(key)

  def __setitem__(self, key, value):
    return self._setitem_impl(key, value)

  def __neg__(self): # -self
    return self._neg_impl()

  def __radd__(self, other): # other + self
    return self._radd_impl(other)

  def __sub__(self, other): # self - other
    return self._sub_impl(other)

  def __rsub__(self, other): # other - self
    return self._rsub_impl(other)

  def __rmul__(self, other): # other * self
    return self._rmul_impl(other)

  def __truediv__(self, other): # self / other
    return self._truediv_impl(other)

  def __rtruediv__(self, other): # other / self
    return self._rtruediv_impl(other)

  def reshape(self, *new_shape):
    return self._reshape_impl(*new_shape)

  def transpose(self, *axes):
    return self._transpose_impl(*axes)
  
  def swapaxes(self, axis1, axis2):
    return self._swapaxes_impl(axis1, axis2)
    
  @property
  def shape(self):
    return self._shape_impl()