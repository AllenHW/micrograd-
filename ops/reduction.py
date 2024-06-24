import numpy as np

from utils import expand_reduced_dims
from base import ReduceOp
from ..tensor.base import Tensor


class Sum(ReduceOp):
  OP = 'sum'

  def forward(self):
    data = np.sum(self.a.data, axis=self.axis, keepdims=self.keepdims)
    self.out = Tensor(data, (self.a,), self)

    return self.out

  def backward(self):
    self.a.grad += expand_reduced_dims(self.out.grad, self.a.shape, 
                                      self.axis, self.keepdims)


class Mean(ReduceOp):
  OP = 'mean'

  def forward(self):
    data = np.mean(self.a.data, axis=self.axis, keepdims=self.keepdims)
    self.out = Tensor(data, (self.a,), self)

    return self.out

  def backward(self):
    if self.axis is None:
      size = np.prod(self.a.shape)
    else:
      axis = (self.axis,) if isinstance(self.axis, int) else self.axis
      size = np.prod([self.a.shape[dim] for dim in axis])
    
    self.a.grad += expand_reduced_dims(self.out.grad / size, self.a.shape, 
                                      self.axis, self.keepdims)


class Max(ReduceOp):
  OP = 'max'

  def forward(self):
    data = np.max(self.a.data, axis=self.axis, keepdims=self.keepdims)
    self.out = Tensor(data, (self.a,), self)
    
    return self.out

  def backward(self):
    indices = np.argmax(self.a.data, axis=self.axis, keepdims=True)
    grad = expand_reduced_dims(
      self.out.grad, indices.shape,
      self.axis, self.keepdims
    )
    curr_grad = np.take_along_axis(self.a.grad, indices, axis=self.axis)
    updated_grad = grad + curr_grad

    np.put_along_axis(self.self.a.grad, indices, updated_grad, axis=self.axis)


class Min(ReduceOp):
  OP = 'min'

  def forward(self):
    self.out = Tensor(np.min(self.a.data, axis=self.axis, keepdims=self.keepdims), 
                      (self.a,), self)
    return self.out

  def backward(self):
    indices = np.argmin(self.a.data, axis=self.axis, keepdims=True)
    grad = expand_reduced_dims(
      self.out.grad, indices.shape,
      self.axis, self.keepdims
    )
    curr_grad = np.take_along_axis(self.a.grad, indices, axis=self.axis)
    updated_grad = grad + curr_grad

    np.put_along_axis(self.self.a.grad, indices, updated_grad, axis=self.axis)
