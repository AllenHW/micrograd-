import numpy as np
from typing import List
from .ops import Add, Subtract, Multiply, Divide, Pow, Matmul, Index

class ndarray:
    def __init__(self, data: np.ndarray | List, _prev=(), _op=None):
        self.data = np.array(data)
        self._prev = set(_prev)
        self._op = _op
        self.grad = None

    def __repr__(self):
        return f"ndarray(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):  # self + other
        op = Add(self, other)
        return op.forward()
    
    __radd__ = __add__
    
    def __sub__(self, other): # self - other
        op = Subtract(self, other)
        return op.forward()
    
    def __rsub__(self, other): # other - self
        return other - self

    def __mul__(self, other):  # self * other
        op = Multiply(self, other)
        return op.forward()
    
    __rmul__ = __mul__

    def __truediv__(self, other): # self / other
        op = Divide(self, other) 
        return op.forward()

    def __rtruediv__(self, other): # other / self
        return other / self
    
    def __pow__(self, other): # self ** other
        op = Pow(self, other)
        return op.forward()
    
    def __matmul__(self, other):
        op = Matmul(self, other)
        return op.forward()

    def __getitem__(self, key):
        op = Index(self, key)
        return op.forward()

    def __setitem__(self, key, value):
        # Implement setitem logic here
        pass

    def __neg__(self): # -self
        # Implement negation logic here
        pass

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

        self.grad = np.ones_like(self.data)
        for v in topo:
            v._backward()

    def _backward(self):
        if self._op is not None:
            # compute gradient for prev nodes
            for t in self._prev:
                if t.grad is None:
                    t.grad = np.zeros_like(t.data)
            self._op.backward()

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @classmethod
    def zeros(cls, shape):
        data = np.zeros(shape)
        return cls(data)

    @classmethod
    def ones(cls, shape):
        data = np.ones(shape)
        return cls(data)

    @classmethod
    def randn(cls, *shape):
        data = np.random.randn(*shape)
        return cls(data)

    @classmethod
    def eye(cls, n, m=None):
        data = np.eye(n, m)
        return cls(data)

    @classmethod
    def arange(cls, start, stop=None, step=1):
        data = np.arange(start, stop, step)
        return cls(data)
