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
    
    
    self._cached_indices = None

    self._transpose_axes = None
    self._transpose_shape = None
    self._nested_shape = None

  def forward(self):
    if self.x.ndim == 0 or self.x.size == 0:
      data = self.op(self.x.data, axis=self.axis, keepdims=self.keepdims)
    else:
      if not isinstance(self.axis, (int, tuple, list, type(None))):
        raise ValueError(f'axis has wrong type {type(self.axis)}')
      
      if isinstance(self.axis, int):
        axis = (self.axis,)
      elif self.axis is None:
        axis = list(range(self.x.ndim))
      else:
        axis = list(self.axis)
      
      self._transpose_axes = axis
      self._flat_shape = [-1]
      self._nested_shape = [self.x.shape[i] for i in axis]
      out_shape = []
      for i, dim in enumerate(self.x.shape):
        if i in axis:
          if self.keepdims:
            out_shape.append(1)
        else:
          self._transpose_axes.append(i)
          self._flat_shape.append(dim)
          self._nested_shape.append(dim)
          out_shape.append(dim)

      x_reshaped = self.x.data.transpose(self._transpose_axes)\
                              .reshape(self._flat_shape)

      indices = self.argop(x_reshaped, axis=0, keepdims=True)
      self._cached_indices = indices
      data = np.take_along_axis(x_reshaped, indices, axis=0)
      data = data.reshape(out_shape)

    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    if self.x.ndim == 0 or self.x.size == 0:
      self.x.grad += self.out.grad
    else:
      curr_grad_reshaped = self.x.grad.transpose(self._transpose_axes)\
                                      .reshape(self._flat_shape)
      grad_reshaped = expand_reduced_dims(
        self.out.grad, self.x.shape,
        self.axis, self.keepdims
      ).transpose(self._transpose_axes).reshape(self._flat_shape)

      curr_grad = np.take_along_axis(curr_grad_reshaped, self._cached_indices, axis=0)
      updated_grad = grad_reshaped + curr_grad
      np.put_along_axis(curr_grad_reshaped, self._cached_indices, updated_grad, axis=0)
      
      inv_axes = np.argsort(self._transpose_axes)
      self.x.grad = curr_grad_reshaped.reshape(self._nested_shape).transpose(inv_axes)


class Max(_MinMaxOp):
  OP = 'max'

class Min(_MinMaxOp):
  OP = 'min'
