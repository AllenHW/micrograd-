import numpy as np
from collections import Counter

from utils import gather_grad
from base import BinaryOp
from ..tensor.base import Tensor


class Matmul(BinaryOp):
  OP = '@'

  def forward(self):
    self.out = Tensor(self.a.data @ self.b.data, (self.a, self.b), self)
    return self.out

  def backward(self):
    if len(self.a.shape >= 2) and (self.b.shape) >= 2:
      # (..., i, k) @ (..., k, j) -> (..., i, j)
      self.a.grad += gather_grad(
        self.out.grad @ np.swapaxes(self.b.data, -1, -2),
        self.a.shape
      )
      self.b.grad += gather_grad(
        np.swapaxes(self.a.data, -1, -2) @ self.out.grad,
        self.b.shape
      )
    elif len(self.b.shape) == 1:
      # (..., i, k) @ (k,) -> (..., i)
      self.a.grad += self.out.grad[:, np.newaxis] * self.b.data
      self.b.grad += gather_grad(
        self.a.data * self.out.grad[:, np.newaxis],
        self.b.shape
      )
    elif len(self.data.shape) == 1:
      # (k,) @ (..., k, j) -> (..., j)
      self.a.grad += gather_grad(
        np.swapaxes(self.b.data, -1, -2) @ self.out.grad[:, np.newaxis],
        self.a.shape
      )
      self.b.grad += np.swapaxes(self.a.data * self.out.grad[:, np.newaxis], -1, -2)


class Einsum():
  OP = 'einsum'

  def __init__(self, subscripts, *operands):
    self.operands = operands
    self.subscripts = subscripts
    self.out = None

    self.in_labels, self.out_label = self._parse_subscripts()
    assert len(self.in_labels) == len(self.operands)

  def _parse_subscripts(self):
    split = self.subscripts.split('->')
    if len(split) == 1:
      in_labels = [x.strip() for x in self.subscripts.split(',')]
      char_counts = {}
      for label in in_labels:
        for c in label:
          char_counts[c] = char_counts.get(c, 0) + 1
      for c, count in char_counts.items():
        if count > 1:
          del char_counts[c]
      out_label = ''.join(sorted(char_counts.keys()))
    else:
      assert len(split) == 2
      in_labels, out_label = split
      out_label = out_label.strip()
      in_labels = [x.strip() for x in in_labels.split(',')]

    assert len(in_labels) == len(self.operands)
    return in_labels, out_label

  def forward(self):
    self.out = Tensor(
      np.einsum(self.subscripts, *self.operands), 
      self.operands, self
    )
    return self.out

  def _handle_repeated_dims(label, operand):
    char_counts = Counter(label)
    out_label = ''
    indices = []
    for i, c in enumerate(label):
      # repeated dimensions will be diagonalized. 
      if char_counts[c] > 1:
        indices.append(range(operand.shape[i]))
      else:
        indices.append(slice(None))
      # include the repeated dimensions only once in the output label
      if c not in out_label:
        out_label += c

    return out_label, indices

  def backward(self):
    for i, operand in enumerate(self.operands):
      label = self.in_labels[i]

      subscripts = f'{self.out_label}'
      operands = [self.out.grad]
      for j in range(len(self.operands)):
        if j != i:
          subscripts += f',{self.input_labels[j]}'
          operands.append(self.operands[j])

      out_label, indices = self._handle_repeated_dims(label, operand)
      
      subscripts = f'{subscripts} -> {out_label}'
      operand.grad[indices] = np.einsum(subscripts, *operands)
