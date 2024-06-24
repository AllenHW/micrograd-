import numpy as np

from utils import gather_grad, log_wo_warning, div_wo_warning
from base import UnaryOp
from ..tensor.base import Tensor


class Log(UnaryOp):
  OP = 'log'

  def forward(self):
    data = log_wo_warning(self.a.data)
    self.out = Tensor(data, (self.a,), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(
      div_wo_warning(1, self.a.data) * self.out.grad, 
      self.a.shape
    )


class Exp(UnaryOp):
  OP = 'exp'

  def forward(self):
    data = np.exp(self.a.data)
    self.out = Tensor(data, (self.a,), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(
      self.out.data * self.out.grad, 
      self.a.shape
    )


class Sin(UnaryOp):
  OP = 'sin'

  def forward(self):
    data = np.sin(self.a.data)
    self.out = Tensor(data, (self.a,), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(np.cos(self.a.data) * self.out.grad, self.a.shape)


class Cos(UnaryOp):
  OP = 'cos'

  def forward(self):
    data = np.cos(self.a.data)
    self.out = Tensor(data, (self.a,), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(-np.cos(self.a.data) * self.out.grad, self.a.shape)


class Tan(UnaryOp):
  OP = 'tan'

  def forward(self):
    data = np.tan(self.a.data)
    self.out = Tensor(data, (self.a,), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(
      div_wo_warning(1, np.cos(self.a.data) ** 2) * self.out.grad, 
      self.a.shape
    )


class Relu(UnaryOp):
  OP = 'relu'

  def forward(self):
    data = np.max(self.a.data, 0)
    self.out = Tensor(data, (self.a,), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(
      (self.a.data > 0) * self.out.grad, 
      self.a.shape
    )


class Tanh(UnaryOp):
  OP = 'tanh'

  def forward(self):
    data = (np.exp(2 * self.a.data) - 1) / (np.exp(2 * self.a.data) + 1)
    self.out = Tensor(data, (self.a,), self)
    return self.out

  def backward(self):
    self.a.grad += gather_grad(
       (1 - (self.out.data ** 2)) * self.out.grad, 
      self.a.shape
    )
