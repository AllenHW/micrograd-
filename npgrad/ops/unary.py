import numpy as np

from .utils import gather_grad, log_wo_warning, div_wo_warning
from .base import UnaryOp


class Log(UnaryOp):
  OP = 'log'

  def forward(self):
    data = log_wo_warning(self.x.data)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    self.x.grad += gather_grad(
      div_wo_warning(1, self.x.data) * self.out.grad, 
      self.x.shape
    )


class Exp(UnaryOp):
  OP = 'exp'

  def forward(self):
    data = np.exp(self.x.data)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    self.x.grad += gather_grad(
      self.out.data * self.out.grad, 
      self.x.shape
    )


class Sin(UnaryOp):
  OP = 'sin'

  def forward(self):
    data = np.sin(self.x.data)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    self.x.grad += gather_grad(np.cos(self.x.data) * self.out.grad, self.x.shape)


class Cos(UnaryOp):
  OP = 'cos'

  def forward(self):
    data = np.cos(self.x.data)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    self.x.grad += gather_grad(-np.sin(self.x.data) * self.out.grad, self.x.shape)


class Tan(UnaryOp):
  OP = 'tan'

  def forward(self):
    data = np.tan(self.x.data)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    self.x.grad += gather_grad(
      div_wo_warning(1, np.cos(self.x.data) ** 2) * self.out.grad, 
      self.x.shape
    )


class Relu(UnaryOp):
  OP = 'relu'

  def forward(self):
    data = np.maximum(self.x.data, 0)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    self.x.grad += gather_grad(
      (self.x.data > 0) * self.out.grad, 
      self.x.shape
    )


class Tanh(UnaryOp):
  OP = 'tanh'

  def forward(self):
    data = (np.exp(2 * self.x.data) - 1) / (np.exp(2 * self.x.data) + 1)
    self.out = self.ndarray_cls(data, (self.x,), self)
    return self.out

  def backward(self):
    self.x.grad += gather_grad(
       (1 - (self.out.data ** 2)) * self.out.grad, 
      self.x.shape
    )
