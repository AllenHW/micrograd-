class Op:
  pass

class BinaryOp:
  def __init__(self, a, b):
    self.a = a
    self.b = b
    self.out = None


class UnaryOp:
  def __init__(self, a):
    self.a = a
    self.out = None


class ReduceOp:
  def __init__(self, operands, axis=None, keepdims=False):
    self.operands = operands
    self.axis = axis
    self.keepdims = keepdims
    self.out = None
