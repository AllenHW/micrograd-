import numpy as np
from collections import Counter, OrderedDict

from .utils import gather_grad
from .base import BinaryOp, Op


class Matmul(BinaryOp):
  OP = '@'

  def forward(self):
    data = self.a.data @ self.b.data
    self.out = self.ndarray_cls(data, (self.a, self.b), self)

    return self.out

  def backward(self):
    if self.a.ndim >= 2 and self.b.ndim >= 2:
      # (..., i, k) @ (..., k, j) -> (..., i, j)
      self.a.grad += gather_grad(
        self.out.grad @ np.swapaxes(self.b.data, -1, -2),
        self.a.shape
      )
      self.b.grad += gather_grad(
        np.swapaxes(self.a.data, -1, -2) @ self.out.grad,
        self.b.shape
      )
    elif self.b.ndim == 1:
      # (..., i, k) @ (k,) -> (..., i)
      self.a.grad += self.out.grad[..., np.newaxis] * self.b.data
      self.b.grad += gather_grad(
        self.a.data * self.out.grad[..., np.newaxis],
        self.b.shape
      )
    elif self.a.ndim == 1:
      # (k,) @ (..., k, j) -> (..., j)
      self.a.grad += gather_grad(
        np.swapaxes(self.b.data, -1, -2) * self.out.grad[..., np.newaxis],
        self.a.shape
      )
      self.b.grad += np.swapaxes(self.a.data * self.out.grad[..., np.newaxis], -1, -2)


class Einsum(Op):
  OP = 'einsum'

  def __init__(self, subscripts, *operands):
    self.operands = operands
    self.subscripts = subscripts
    self.out = None

    self.ndarray_cls = self.operands[0].__class__

    self.in_labels, self.out_label = self._parse_subscripts()
    assert len(self.in_labels) == len(self.operands)

  def _parse_subscripts(self):
    split = self.subscripts.split('->')

    in_labels = [x.strip() for x in split[0].split(',')]
    # implicit case
    if len(split) == 1:
      char_counts = Counter(c for label in in_labels for c in label)
      duplicate_chars = []
      for c, count in char_counts.items():
        if count > 1:
          duplicate_chars.append(c)
      for c in duplicate_chars:
        del char_counts[c]
      out_label = ''.join(sorted(char_counts.keys()))
    # explict case
    else:
      assert len(split) == 2
      out_label = split[1].strip()

    assert len(in_labels) == len(self.operands)
    return in_labels, out_label

  def forward(self):
    data = np.einsum(self.subscripts, *[x.data for x in self.operands])
    self.out = self.ndarray_cls(data, self.operands, self)

    return self.out

  def _handle_repeated_dims(self, label, input_operand):
    char_counts = Counter(label)

    non_repeated_dims = ''
    repeated_dims = ''
    repeated_sizes = OrderedDict()
    for i, c in enumerate(label):
      if char_counts[c] == 1:
        non_repeated_dims += c
      else:
        if c not in repeated_sizes:
          repeated_sizes[c] = range(input_operand.shape[i])
        # include the repeated dimensions only once in the output label
        if c not in repeated_dims:
          repeated_dims += c

    mesh_idx = np.ix_(*repeated_sizes.values())
    mesh_idx_lookup = {}
    for i, c in enumerate(repeated_sizes.keys()):
      mesh_idx_lookup[c] = mesh_idx[i]

    first_repeated_dims = None
    repeated_dims_contiguous = True
    indices = []
    for i, c in enumerate(label):
      if char_counts[c] == 1:
        indices.append(slice(None))
      if char_counts[c] > 1:
        indices.append(mesh_idx_lookup[c])
        
        if first_repeated_dims is None:
          first_repeated_dims = i
        else:
          if type(indices[-2]) != np.ndarray:
            repeated_dims_contiguous = False

    if first_repeated_dims is None:
      processed_label = non_repeated_dims
    else:
      if repeated_dims_contiguous:
        processed_label = non_repeated_dims[:first_repeated_dims] + \
                        repeated_dims + non_repeated_dims[first_repeated_dims:]
      else:
        processed_label = repeated_dims + non_repeated_dims

    return processed_label, tuple(indices)

  def _self_reduced_dims(self, i):
    label = self.in_labels[i]
    other_labels = self.in_labels[:i-1] if i > 0 else []
    other_labels += self.in_labels[i+1:] if i < len(self.in_labels) - 1 else []
    other_labels += [self.out_label]

    all_dims = set(''.join(other_labels))
    self_reduced_dims = ''
    for i, c, in enumerate(label):
      if c not in all_dims and c not in self_reduced_dims:
        self_reduced_dims += c

    return self_reduced_dims

  def backward(self):
    for i, operand in enumerate(self.operands):
      label = self.in_labels[i]

      processed_label, indices = self._handle_repeated_dims(label, operand)
      self_reduced_dims = self._self_reduced_dims(i)

      out_label = self.out_label + self_reduced_dims
      out_grad_shape = self.out.grad.shape + (1,) * len(self_reduced_dims)
      out_grad = np.reshape(self.out.grad, out_grad_shape)

      subscripts = f'{out_label}'
      operands = [out_grad]
      for j in range(len(self.operands)):
        if j != i:
          subscripts += f',{self.in_labels[j]}'
          operands.append(self.operands[j].data)

      subscripts = f'{subscripts} -> {processed_label}'
      operand.grad[indices] += np.einsum(subscripts, *operands)