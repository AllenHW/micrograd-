import numpy as np

from .utils import expand_reduced_dims
from .base import ReduceOp


class Sum(ReduceOp):
  OP = 'sum'

  def forward(self):
    data = np.sum(self.x.data, axis=self.axis, keepdims=self.keepdims)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    self.x.grad += expand_reduced_dims(self.out.grad, self.x.shape, 
                                      self.axis, self.keepdims)


class Mean(ReduceOp):
  OP = 'mean'

  def forward(self):
    data = np.mean(self.x.data, axis=self.axis, keepdims=self.keepdims)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    if self.axis is None:
      size = np.prod(self.x.shape)
    else:
      axis = (self.axis,) if isinstance(self.axis, int) else self.axis
      size = np.prod([self.x.shape[dim] for dim in axis])
    
    self.x.grad += expand_reduced_dims(self.out.grad / size, self.x.shape, 
                                      self.axis, self.keepdims)


class _MinMaxOp(ReduceOp):
  OP = None

  def __init__(self, x, axis=None, keepdims=False):
    assert self.OP in ('min', 'max')
    super().__init__(x, axis, keepdims)
    self.argop = np.argmax if self.OP == 'max' else np.argmin
    self.op = np.max if self.OP == 'max' else np.min

  def forward(self):
    data = self.op(self.x.data, axis=self.axis, keepdims=self.keepdims)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    if self.x.ndim == 0 or self.x.size == 0:
      self.x.grad += self.out.grad
    else:
      if isinstance(self.axis, int):
        axis = (self.axis,)
      elif self.axis is None:
        axis = list(range(self.x.ndim))
      else:
        axis = list(self.axis)

      transpose_axes = axis
      flat_shape = [-1]
      nested_shape = [self.x.shape[i] for i in axis]
      for i, dim in enumerate(self.x.shape):
        if i not in axis:
          transpose_axes.append(i)
          flat_shape.append(dim)
          nested_shape.append(dim)

      grad_reshaped = self.out.grad.reshape(flat_shape)

      x_reshaped = self.x.data.transpose(transpose_axes).reshape(flat_shape)
      indices = self.argop(x_reshaped, axis=0, keepdims=True)
      curr_grad_reshaped = self.x.grad.transpose(transpose_axes).reshape(flat_shape)
      curr_grad = np.take_along_axis(curr_grad_reshaped, indices, axis=0)

      updated_grad = grad_reshaped + curr_grad
      np.put_along_axis(curr_grad_reshaped, indices, updated_grad, axis=0)
      
      inv_axes = np.argsort(transpose_axes)
      self.x.grad = curr_grad_reshaped.reshape(nested_shape).transpose(inv_axes)


class Max(_MinMaxOp):
  OP = 'max'

class Min(_MinMaxOp):
  OP = 'min'
