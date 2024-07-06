import numpy as np
from itertools import accumulate
from .base import Op

class Repeat(Op):
  OP = 'repeat'

  def __init__(self, x, repeats, axis=None):
    self.x = x 
    self.repeats = repeats
    self.axis = axis

    self.ndarray_cls = self.x.__class__

  def forward(self):
    data = np.repeat(self.x.data, self.repeats, self.axis)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out
  
  def backward(self):
    if self.axis is None:
      self.x.grad += np.sum(self.out.grad.reshape(self.x.shape + (-1,)), axis=-1)
    else:
      if isinstance(self.repeats, int):
        indices_or_sections = self.x.shape[self.axis]
        if indices_or_sections == 0:
          return 
      elif isinstance(self.repeats, list):
        indices_or_sections = list(accumulate(self.repeats))[:-1]

      for i, section in enumerate(np.split(self.out.grad, indices_or_sections, axis=self.axis)):
        idx = (np.s_[:],)*self.axis + (i,) + (np.s_[:],)*(self.x.ndim-self.axis-1)
        self.x.grad[idx] += np.sum(section, axis=self.axis)


class Tile(Op):
  OP = 'tile'

  def __init__(self, x, reps):
    self.x = x
    self.reps = reps

    self.ndarray_cls = self.x.__class__

  def forward(self):
    data = np.tile(self.x.data, self.reps)
    self.out = self.ndarray_cls(data, (self.x,), self)

    return self.out
  
  def backward(self):
    if len(self.reps) > self.x.ndim:
      expanded_dims = tuple(range(len(self.reps) - self.x.ndim))
      x_broadcasted = np.expand_dims(self.x.data, expanded_dims)
    else:
      expanded_dims = ()
      x_broadcasted = self.x.data

    untiled_dims = x_broadcasted.ndim - len(self.reps)
    shape = []
    reduce_axes = []
    for i in range(x_broadcasted.ndim):
      if i < untiled_dims:
        shape.append(x_broadcasted.shape[i])
      else:
        reduce_axes.append(len(shape))
        shape += [self.reps[i - untiled_dims], x_broadcasted.shape[i]]

    grad_reshaped = np.reshape(self.out.grad, shape)
    self.x.grad += np.sum(grad_reshaped, axis=tuple(reduce_axes), keepdims=False)\
                    .squeeze(expanded_dims)
