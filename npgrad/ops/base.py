class Op:
  pass

class BinaryOp:
  def __init__(self, a, b):
    self.a = a
    self.b = b
    self.out = None

    self.ndarray_cls = a.__class__


class UnaryOp:
  def __init__(self, x):
    self.x = x
    self.out = None

    self.ndarray_cls = x.__class__



class ReduceOp:
  def __init__(self, x, axis=None, keepdims=False):
    self.x = x
    self.axis = axis
    self.keepdims = keepdims
    self.out = None

    self.ndarray_cls = x.__class__


class PoolingOp:
  def __init__(self, x, k, s=1, p=1, d=1):
    self.x = x
    self.k = k
    self.s = s
    self.p = p
    self.d = d

    self.ndarray_cls = x.__class__
