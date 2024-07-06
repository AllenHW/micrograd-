import numpy as np

def gather_grad(grad, target_shape):
  if grad.shape == target_shape:
    return grad
  
  if grad.size == 0:
    return np.zeros(target_shape)

  assert len(target_shape) <= len(grad.shape)
  ndim_gap = len(grad.shape) - len(target_shape)

  gather_dims = []
  squeeze_dims = []
  for i, dim in enumerate(grad.shape):
    if i < ndim_gap: 
      squeeze_dims.append(i)
      continue
    assert target_shape[i - ndim_gap] <= dim
    if target_shape[i - ndim_gap] == 1 and dim > 1:
      gather_dims.append(i)


  gather_dims = tuple(gather_dims)
  squeeze_dims = tuple(squeeze_dims)
  coalesced = np.sum(grad, axis=squeeze_dims + gather_dims, keepdims=True)
  squeezed = np.squeeze(coalesced,  axis=squeeze_dims)
  
  return squeezed

def expand_reduced_dims(x, original_shape, axis, keepdims):
  if keepdims:
    return x

  if axis is None:
    axis = range(len(original_shape))
  elif isinstance(axis, int):
    axis = (axis,)
  
  shape = list(original_shape)
  for dim in axis:
    shape[dim] = 1
  
  return x.reshape(shape)


def log_wo_warning(x):
  with np.errstate(all='ignore'):
    return np.log(x)
  
def div_wo_warning(x, y):
  with np.errstate(all='ignore'):
    return x / y

def pow_wo_warning(x, y):
  with np.errstate(all='ignore'):
    return x ** y