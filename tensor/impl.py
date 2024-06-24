import numpy as np
from base import Tensor
from ops import (
  Add, Subtract, Multiply, Divide, Pow,  # noqa: F401
  Slice, Assign, MaskedFill, Lookup,  # noqa: F401
  Sum, Mean, Max, Min,  # noqa: F401
  Tranpose,  Reshape, Swapaxes, Ravel, Squeeze, Expanddims, Concat,  # noqa: F401
  Matmul, Einsum,  # noqa: F401
  Log, Exp, Sin, Cos, Tan, Relu, Tanh  # noqa: F401
)


class TensorImpl(Tensor):
  
  def _add_impl(self, other):
    pass

  def _mul_impl(self, other):
    pass
  
  def _pow_impl(self, other):
    pass
    
  def _matmul_impl(self, other):
    pass

  def _getitem_impl(self, key):
    pass

  def _setitem_impl(self, key, value):
    pass

  # def _reshape_impl(*new_shape):

  # def _transpose_impl(*axes):
  
  # def _swapaxes_impl(axis1, axis2):
    
  def _shape_impl(self):
    return self.data.shape

  def _neg_impl(self, ):
    pass

  def _radd_impl(self, other):
    pass

  def _sub_impl(self, other):
    pass

  def _rsub_impl(self, other):
    pass

  def _rmul_impl(self, other):
    pass

  def _truediv_impl(self, other):
    pass

  def _rtruediv_impl(self, other):
    pass

  def _backward(self):
    if self.op is not None:
      # compute gradient for prev nodes
      for t in self._prev:
        if t.grad is None:
          t.grad = np.zeros_like(t.data.shape)
      self.op.backward()

  def backward(self):
    visited = set()

    topo = []
    def build_topo(v):
      if v not in visited:
          visited.add(v)
          for child in v._prev:
              build_topo(child)
          topo.append(v)
    build_topo(self)
    topo = reversed(topo)

    self.grad = np.ones_like(self.data.shape)
    for v in topo:
      v._backward()