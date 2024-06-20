import math

class Scalar:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self._prev = set(_children)
    self._backward = lambda: None
    self.grad = 0
    self._op = _op
    self.label = label
  
  def __add__(self, other):  # self + other
    other = other if isinstance(other, Scalar) else Scalar(other)
    out = Scalar(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    
    return out
  
  def __mul__(self, other):  # self * other
    other = other if isinstance(other, Scalar) else Scalar(other)
    out = Scalar(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward
    return out
  
  def __pow__(self, other): # self ** other
    assert isinstance(other, int) or isinstance(other, float)
    out = Scalar(self.data ** other.data, (self, other), '**')

    def _backward():
      self.grad += other.data * self.data**(other.data - 1) * out.grad
      # other.grad = math.log(self.data) * math.exp(other.data) * out.grad

    out._backward = _backward
    return out

  def relu(self): # relu(self)
    out = Scalar(self.data if self.data > 0 else 0, (self,), 'ReLU')

    def _backward():
      self.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x)-1) / (math.exp(2*x)+1)
    out = Scalar(t, (self,), 'tanh')

    def _backward():
      self.grad += (1 - (out.data ** 2)) * out.grad

    out._backward = _backward
    return out

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

    self.grad = 1.0
    for v in topo:
      v._backward()

  def __neg__(self): # -self
    return self * -1

  def __radd__(self, other): # other + self
    return self + other

  def __sub__(self, other): # self - other
    return self + (-other)

  def __rsub__(self, other): # other - self
    return self + (-other)

  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __rtruediv__(self, other): # other / self
    return  self * other**-1

  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
