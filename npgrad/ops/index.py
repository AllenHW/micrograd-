import numpy as np
from .base import Op

class Index(Op): 
  OP = 'index'
  
  def __init__(self, x, index):
    self.x = x
    self.index = index
    self.out = None
  
    self.ndarray_cls = self.x.__class__

  def forward(self):
    data = self.x.data[self.index]
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    np.add.at(self.x.grad, self.index, self.out.grad)


class MaskedFill(Op):
  OP = 'maskedfill'

  def __init__(self, x, mask, value):
    self.x = x
    self.mask = mask
    self.value = value
    self.out = None

    self.ndarray_cls = self.x.__class__
  
  def forward(self):
    data = self.x.data
    data[self.mask.data] = self.value
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out

  def backward(self):
    pass

class EmbeddingLookup(Op):
  OP = 'embedding_lookup'
  pass
