import numpy as np
from .base import PoolingOp


def _split_hw(x):
  return (x, x) if isinstance(x, (int, float)) else (x[0], x[1])

def check_shape(shape, k, s, p, d):
  _, _, H, W = shape
  kh, kw = _split_hw(k)
  sh, sw = _split_hw(s)
  ph, pw = _split_hw(p)

  dh, dw = _split_hw(d)

  dilated_kh = kh + (kh-1) * (dh-1)
  dilated_kw = kw + (kw-1) * (dw-1)
  assert (H + 2 * ph - dilated_kh) % sh == 0
  assert (W + 2 * pw - dilated_kw) % sw == 0

  out_H = (H + 2 * ph - dilated_kh) // sh + 1
  out_W = (W + 2 * pw - dilated_kw) // sw + 1

  return out_H, out_W

def im2cols_indices(shape, k, s=1, p=1, d=1):
  _, C, _, _ = shape
  kh, kw = _split_hw(k)
  sh, sw = _split_hw(s)
  dh, dw = _split_hw(d)
  out_H, out_W = check_shape(shape, k, s, p, d)

  i0 = np.repeat(np.arange(0, kh * dh, dh), kw)
  i0 = np.tile(i0, C)

  i1 = sh * np.repeat(np.arange(out_H), out_W)
  j0 = np.tile(np.arange(0, kw * dw, dw), kh)
  j0 = np.tile(j0, C)
  j1 = sw * np.tile(np.arange(out_W), out_H)

  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)
  k = np.repeat(np.arange(C), kh * kw).reshape(-1, 1)

  return k, i, j

def im2cols(x, k, s=1, p=1, d=1, pad_value=0):
  ph, pw = _split_hw(p)

  x_padded = np.pad(x, ((0,0),(0,0),(ph, ph),(pw, pw)), 
                    mode='constant', constant_values=pad_value)
  k, i, j = im2cols_indices(x.shape, k, s, p, d)

  cols = x_padded[:, k, i, j]

  return cols


def cols2im(cols, shape, k, s=1, p=1, d=1):
  B, out_C, H, W = shape
  ph, pw = _split_hw(p)

  k, i, j = im2cols_indices(shape, k, s, p, d)
  
  padded_shape = (B, out_C, H + 2 * ph, W + 2 * pw)
  im_padded = np.zeros(padded_shape)
  np.add.at(im_padded, (slice(None), k, i, j), cols)

  h_slice = slice(ph, -ph, 1) if ph > 0 else slice(None)
  w_slice = slice(pw, -pw, 1) if pw > 0 else slice(None)
  im = im_padded[:, :, h_slice, w_slice]

  return im


class MaxPool2D(PoolingOp):
  OP = 'maxpool2d'

  def __init__(self, x, k, s=1, p=1, d=1):
    super().__init__(x, k, s, p, d)
    self.out_H, self.out_W = check_shape(self.x.shape, k, s, p, d)
    self._cached_x_col = None
    self._cached_index = None

  def forward(self):
    kh, kw = _split_hw(self.k)
    B, in_C, _, _ = self.x.shape

    # B x (in_C * kh * kw) x (out_H * out_W)
    x_col = im2cols(self.x.data, self.k, self.s, self.p, self.d, -np.inf)
    x_col = x_col.reshape(B, in_C, kh * kw, self.out_H * self.out_W)
    self._cached_x_col = x_col

    # B x in_C x (kh * kw) x (out_H * out_W)
    indices = np.argmax(x_col, axis=-2, keepdims=True)
    self._cached_indices = indices

    y = np.take_along_axis(x_col, indices, axis=-2)\
          .reshape(B, in_C, self.out_H, self.out_W)
    self.out = self.ndarray_cls(y, (self.x, ), self)

    return self.out
  
  def backward(self):
    kh, kw = _split_hw(self.k)
    B, in_C, _, _ = self.x.shape

    # B x in_C x (kh * kw) x (out_H * out_W)
    x_grad_col = np.zeros_like(self._cached_x_col)
    out_col = self.out.grad.reshape(B, in_C, 1, -1)
    np.put_along_axis(x_grad_col, self._cached_indices, out_col, axis=-2)

    x_grad_col = x_grad_col.reshape(B, in_C * kh * kw, -1)
    x_grad = cols2im(x_grad_col, self.x.shape, self.k, self.s, self.p, self.d)

    self.x.grad += x_grad


class AvgPool2D(PoolingOp):
  OP = 'avgpool2d'

  def __init__(self, x, k, s=1, p=1, d=1):
    super().__init__(x, k, s, p, d)
    self.out_H, self.out_W = check_shape(self.x.shape, k, s, p, d)
    self._cached_x_col = None

  def forward(self):
    kh, kw = _split_hw(self.k)
    B, in_C, _, _ = self.x.shape

    # B x (in_C * kh * kw) x (out_H * out_W)
    x_col = im2cols(self.x.data, self.k, self.s, self.p, self.d)
    x_col = x_col.reshape(B, in_C, kh * kw, self.out_H * self.out_W)
    self._cached_x_col = x_col

    y = np.mean(x_col, axis=-2).reshape(B, in_C, self.out_H, self.out_W)
    self.out = self.ndarray_cls(y, (self.x,), self)

    return self.out
  
  def backward(self):
    kh, kw = _split_hw(self.k)
    B, in_C, _, _ = self.x.shape

    # B x in_C x (kh * kw) x (out_H * out_W)
    out_col = self.out.grad.reshape(B, in_C, -1, self.out_H * self.out_W)
    sizes = np.full((1, 1, kh * kw,1), kh * kw)

    x_grad_col = out_col / sizes
    x_grad_col = x_grad_col.reshape(B, in_C * kh * kw, self.out_H * self.out_W)
    x_grad = cols2im(x_grad_col, self.x.shape, self.k, self.s, self.p, self.d)

    self.x.grad += x_grad


# https://agustinus.kristia.de/techblog/2016/07/16/convnet-conv-layer/
class Conv2D(PoolingOp):
  OP = 'conv2d'

  def __init__(self, x, W, b, s=1, p=1, d=1):
    k = W.shape[-2:]
    super().__init__(x, k, s, p, d)

    self.W = W
    self.b = b
    self.out_H, self.out_W = check_shape(self.x.shape, k, s, p, d)

    self._cached_x_col = None

  def forward(self):
    # B x (in_C * kh * kw) x (out_H * out_W)
    x_col = im2cols(self.x.data, self.k, self.s, self.p, self.d)
    self._cached_x_col = x_col

    out_C, _ , _, _ = self.W.shape
    # 1 x out_C x (in_C * kh * kw) 
    W_col = self.W.data.reshape(out_C, -1)[np.newaxis, ...]
    # 1 x out_C x 1
    b = self.b.data.reshape(1, out_C, 1)

    # B x out_C x (out_H * out_W)
    y = (W_col @ x_col + b).reshape(-1, out_C, self.out_H, self.out_W)
    self.out = self.ndarray_cls(y, (self.W, self.x, self.b), self)

    return self.out
  
  def backward(self):
    out_C, in_C, kh, kw = self.W.shape

    # B x out_C x (out_H * out_W)
    out_col = self.out.grad.reshape(-1, out_C, self.out_H * self.out_W) 
    # 1 x out_C x (in_C * kh * kw) 
    W_col = self.W.data.reshape(out_C, -1)[np.newaxis, ...]
    # B x (in_C * kh * kw) x (out_H * out_W)
    x_col = self._cached_x_col

    # B x out_C x (in_C * kh * kw)
    W_grad_col = out_col @ np.swapaxes(x_col, 1, 2)
    W_grad_col = np.sum(W_grad_col, axis=0)
    # in_C x out_C x kh x kw
    W_grad = W_grad_col.reshape(out_C, in_C, kh, kw)

    # ... (in_C * kh * kw) x out_C @ ... out_C x (out_H * out_W) 
    # -> B x (in_C * kh * kw) x (out_H * out_W) 
    x_grad_col = np.swapaxes(W_col, 1, 2) @ out_col 
    x_grad = cols2im(x_grad_col, self.x.shape, self.k, self.s, self.p, self.d)

    # out_C
    b_grad = np.sum(self.out.grad, axis=(0, 2, 3))

    self.W.grad += W_grad 
    self.x.grad += x_grad
    self.b.grad += b_grad
