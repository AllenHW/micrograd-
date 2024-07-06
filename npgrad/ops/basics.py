from .utils import gather_grad, log_wo_warning, div_wo_warning, pow_wo_warning
from .base import BinaryOp

class Add(BinaryOp):
  OP = '+'

  def forward(self):
    data = self.a.data + self.b.data
    self.out = self.ndarray_cls(data, (self.a, self.b), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(self.out.grad, self.a.shape)
    self.b.grad += gather_grad(self.out.grad, self.b.shape)

class Subtract(BinaryOp):
  OP = '-'

  def forward(self):
    data = self.a.data - self.b.data
    self.out = self.ndarray_cls(data, (self.a, self.b), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(self.out.grad, self.a.shape)
    self.b.grad += gather_grad(-self.out.grad, self.b.shape)


class Multiply(BinaryOp):
  OP = '*'

  def forward(self):
    data = self.a.data * self.b.data
    self.out = self.ndarray_cls(data, (self.a, self.b), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(self.b.data * self.out.grad, self.a.shape)
    self.b.grad += gather_grad(self.a.data * self.out.grad, self.b.shape)


class Divide(BinaryOp):
  OP = '/'

  def forward(self):
    data = div_wo_warning(self.a.data, self.b.data)
    self.out = self.ndarray_cls(data, (self.a, self.b), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(
      self.out.grad / self.b.data, 
      self.a.shape
    )
    self.b.grad += gather_grad(
      div_wo_warning(-self.a.data * self.out.grad, self.b.data ** 2),
      self.b.shape
    )

class Pow(BinaryOp):
  OP = '^'

  def forward(self):
    data = pow_wo_warning(self.a.data, self.b.data)
    self.out = self.ndarray_cls(data, (self.a, self.b), self)

    return self.out

  def backward(self):
    self.a.grad += gather_grad(
      (self.b.data * pow_wo_warning(self.a.data, self.b.data - 1)) * self.out.grad,
      self.a.shape
    )
    self.b.grad += gather_grad(
      (log_wo_warning(self.a.data) * self.out.data) * self.out.grad,
      self.b.shape
    )
