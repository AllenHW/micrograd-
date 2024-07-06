import npgrad as npg
import numpy as np
import torch
import pytest


@pytest.mark.parametrize("shape", [
  (()),
  ((3,)),
  ((4,5)),
  ((10, 6, 1)),
  ((1, 6, 10)),
  ((0, 5, 3)),
  ((5, 3, 0))
])
def test_unary(shape):
  _test_op(shape, npg.log, torch.log)
  _test_op(shape, npg.exp, torch.exp)
  _test_op(shape, npg.sin, torch.sin)
  _test_op(shape, npg.cos, torch.cos)
  _test_op(shape, npg.tan, torch.tan)
  _test_op(shape, npg.relu, torch.relu)
  _test_op(shape, npg.tanh, torch.tanh)


@pytest.mark.parametrize("shape1, shape2", [
  ((), ()),
  ((3,), (1,)),
  ((4,5), (4,5)),
  ((10, 6, 1), (10, 6, 1)),
  ((1, 6, 10), (1, 6, 10)),
  ((1, 6, 10), (4, 6, 10)),
  ((4, 6, 10), (1, 6, 10)),
  ((4, 6, 10), (4, 1, 10)),
  ((4, 6, 10), (1, 1, 1)),
  ((4, 6, 10), (1, 1, 1, 1)),
  ((7, 4, 6, 10), (4, 1, 1)),
  ((2, 3, 2, 4, 6, 10), (1, 6, 10)),
  ((0, 3, 4), (1, 1, 3, 4)),
  ((5, 0, 4), (1, 1, 4)),
])
def test_basics(shape1, shape2):
  _test_magic_method(shape1, shape2, '__add__')
  _test_magic_method(shape1, shape2, '__radd__')
  _test_magic_method(shape1, shape2, '__sub__')
  _test_magic_method(shape1, shape2, '__rsub__')
  _test_magic_method(shape1, shape2, '__mul__')
  _test_magic_method(shape1, shape2, '__rmul__')
  _test_magic_method(shape1, shape2, '__truediv__')
  _test_magic_method(shape1, shape2, '__rtruediv__')
  _test_magic_method(shape1, shape2, '__pow__')



def _get_possible_axes(shape):
  def _get_sequences(xs):
    rets = [[]]

    for x in xs:
      sub_sequences = _get_sequences(xs - {x})
      for seq in sub_sequences:
        rets.append([x]+seq)
    
    return rets
  
  seqs = _get_sequences(set(range(len(shape))))
  rets = []
  for seq in seqs:
    if len(seq) == 0:
      rets.append(None)
      rets.append(())
    else:
      rets.append(tuple(seq))

  return rets


@pytest.mark.parametrize("shape", [
  (()),
  ((1,)),
  ((3,)),
  ((4,5)),
  ((1,4,5)),
  ((6,8,2)),
  ((6,1,2)),
  ((1,1,1)),
  ((1,5,1)),
  ((5,0,2)),
])
def test_reduction(shape):
  axes = _get_possible_axes(shape)
  for axis in axes:
    if axis == ():
      _a = np.random.randn(*shape)
      a = npg.ndarray(_a)
      c_min = npg.min(a, axis=axis, keepdims=True)
      c_max = npg.max(a, axis=axis, keepdims=True)
      c_mean = npg.mean(a, axis=axis, keepdims=True)
      c_sum= npg.mean(a, axis=axis, keepdims=True)

      assert _allclose(a.data, c_min.data)
      assert _allclose(a.data, c_max.data)
      assert _allclose(a.data, c_mean.data)
      assert _allclose(a.data, c_sum.data)
    else:
      # min
      if 0 not in shape:
        _test_op(shape, npg.min, torch.amin, 
          npg_kwargs={'axis': axis, 'keepdims': True}, 
          torch_kwargs={'dim': axis, 'keepdim': True}, 
        )
        _test_op(shape, npg.min, torch.amin, 
          npg_kwargs={'axis': axis, 'keepdims': False}, 
          torch_kwargs={'dim': axis, 'keepdim': False}, 
        )

        # max
        _test_op(shape, npg.max, torch.amax, 
          npg_kwargs={'axis': axis, 'keepdims': True}, 
          torch_kwargs={'dim': axis, 'keepdim': True}, 
        )
        _test_op(shape, npg.max, torch.amax, 
          npg_kwargs={'axis': axis, 'keepdims': False}, 
          torch_kwargs={'dim': axis, 'keepdim': False}, 
        )

        # mean
        _test_op(shape, npg.mean, torch.mean, 
          npg_kwargs={'axis': axis, 'keepdims': True}, 
          torch_kwargs={'dim': axis, 'keepdim': True}, 
        )
        _test_op(shape, npg.mean, torch.mean, 
          npg_kwargs={'axis': axis, 'keepdims': False}, 
          torch_kwargs={'dim': axis, 'keepdim': False}, 
        )

      # sum
      _test_op(shape, npg.sum, torch.sum, 
        npg_kwargs={'axis': axis, 'keepdims': True}, 
        torch_kwargs={'dim': axis, 'keepdim': True}, 
      )
      _test_op(shape, npg.sum, torch.sum, 
        npg_kwargs={'axis': axis, 'keepdims': False}, 
        torch_kwargs={'dim': axis, 'keepdim': False}, 
      )



@pytest.mark.parametrize("shape, repeats, axis", [
  ((), 3, 0),
  ((), 3, None),
  ((1,), 2, 0),
  ((1,), 2, None),
  ((3,), 3, 0),
  ((3,), 3, None),
  ((3,), [3,4,2], 0),
  ((3,5,4), 3, 0),
  ((3,5,4), 3, 2),
  ((3,5,4), [3,4,2], 0),
  ((3,5,4), [3,4,2,7], 2),
  ((1,5,4), [3], 0),
  ((1,5,4), [3,4,2,7], 2),
  ((1,1,1), [2], 2),
  ((5,0,2), [2,2], 2),
  ((5,0,2), 2, 0),
  ((5,0,2), 2, 1),
])
def test_repeat(shape, repeats, axis):
  if shape == ():
    _a = np.random.randn(*shape)
    a = npg.ndarray(_a)
    c = npg.repeat(a, repeats, axis)
    assert c.shape == (repeats,)
  else:
    _test_op(shape, npg.repeat, torch.repeat_interleave, 
      npg_kwargs={'repeats': repeats, 'axis': axis}, 
      torch_kwargs={'repeats': torch.tensor(np.array(repeats)), 'dim': axis}, 
    )


@pytest.mark.parametrize("shape, reps", [
  ((), (2,)),
  ((), (2,3)),
  ((3,), (2,)),
  ((2, 3), (3, 2)),
  ((2, 3, 4), (2,)),
  ((5, 1, 3), (1, 4, 2)),
  ((3,), (2, 2, 3)),
  ((2, 3, 4), (1, 2, 1)),
  ((3, 4, 5), (1, 1, 1)),
  ((2, 1, 3, 1, 2), (1, 2, 1, 3, 2)),
  ((5, 0, 2), (2, )),
  ((5, 0, 2), (2, 2)),
  ((5, 0, 2), (3, 2, 2)),
  ((5, 0, 2), (4, 3, 2, 2)),
])
def test_tile(shape, reps):
  _test_op(shape, npg.tile, torch.tile, 
    npg_kwargs={'reps': reps}, torch_kwargs={'dims': reps}
  )



@pytest.mark.parametrize("shape, axes", [
  ((2, 3), None),
  ((2, 3), (1, 0)),
  ((2, 3, 4), (2, 0, 1)),
  ((2, 3, 4), (1, 2, 0)),
  ((2, 3, 4, 5), (3, 2, 1, 0)),
  ((2, 3, 4), (0, 2, 1)),
  ((5,), None),
  ((5,), (0,)),
  ((1, 3, 4), (2, 1, 0)),
  ((2, 1, 4), (0, 2, 1)),
  ((2, 3, 4, 5, 6, 7), (5, 4, 3, 2, 1, 0)),
  ((0, 3), (1, 0)),
  ((), None),
])
def test_transpose(shape, axes):
  if axes is None:
    dims = tuple(reversed(range(len(shape))))
  else:
    dims = axes

  _test_op(shape, npg.transpose, torch.permute, 
    npg_kwargs={'axes': axes}, torch_kwargs={'dims': dims}
  )


@pytest.mark.parametrize("shape, axis1, axis2", [
  ((2, 3), 0, 1),
  ((2, 3), 1, 0),
  ((2, 3, 4), 2, 1),
  ((2, 3, 4), 0, 2),
  ((2, 3, 4), 1, 1),
  ((2, 3, 4), -2, -1),
  ((5,), 0, 0),
  ((2, 3, 4, 5, 6, 7), 1, 5),
  ((0, 3), 1, 0),
])
def test_swapaxes(shape, axis1, axis2):
  _test_op(shape, npg.swapaxes, torch.swapaxes, 
    npg_kwargs={'axis1': axis1, 'axis2': axis2},
    torch_kwargs={'axis0': axis1, 'axis1': axis2}
  )



@pytest.mark.parametrize("shape", [
  ((2, 3)),
  ((1, 4)),
  ((5,)),
  ((1,)),
  ((2, 3, 4, 5, 6, 7)),
  ((0, 3)),
  ((0, 3, 5)),
])
def test_ravel(shape):
  _test_op(shape, npg.ravel, torch.ravel)



@pytest.mark.parametrize("shape, axis", [
  ((2, 3), None),
  ((2, 3, 4), None),
  ((1, 3), None),
  ((1, 1), None),
  ((1, 5, 1), None),
  ((1, 3), 0),
  ((1, 1), 0),
  ((1, 1), 1),
  ((1, 1), (0, 1)),
  ((1, 5, 1), (0,)),
  ((1, 5, 1), (0, 2)),
  ((1, 5, 1), (2,)),
])
def test_squeeze(shape, axis):
  if axis is not None:
    _test_op(shape, npg.squeeze, torch.squeeze, 
      npg_kwargs={'axis': axis}, torch_kwargs={'dim': axis}
    )
  else: 
    _test_op(shape, npg.squeeze, torch.squeeze, 
      npg_kwargs={'axis': axis}
    )



@pytest.mark.parametrize("shape, newshape", [
  ((6,), (2, 3)),
  ((2, 3), (6,)),
  ((2, 3), (3, 2)),
  ((6,), (-1, 2)),
  ((2, 3, 4), (-1,)),
  ((2, 3, 4), (6, -1)),
  ((24,), (2, 3, -1)),
  ((4, 3, 2), (-1,)),
  ((24,), (4, 3, 2)),
  ((5, 1, 3), (15,)),
  ((2, 3, 4, 5), (6, 20)),
  ((120,), (2, 3, 4, 5)),
  ((0,), (0, 2)),  # Empty array
  ((1,), ()),  # Reshape to scalar
  ((), (1,)),  # Reshape from scalar
])
def test_reshape(shape, newshape):
  _test_op(shape, npg.reshape, torch.reshape, 
    npg_kwargs={'newshape': newshape}, 
    torch_kwargs={'shape': newshape}, 
  )


@pytest.mark.parametrize("shape, axis", [
  ((5,), 0,),
  ((5,), 1,),
  ((5,), -1),
  ((2, 3), 0), 
  ((2, 3), -1),
  ((2, 3, 4), 0),
  ((2, 3, 4), 3),
  ((2, 3, 4, 5), 2),
  ((3, 2, 1), 0),
  ((3, 2, 1), 2),
  ((), 0),
  ((2, 3), (0, 2)),
  ((2, 3), (1, -1)),
  ((2, 3, 4, 5), (2, 3, 4)),
  ((2, 3, 4, 5), (-1, 1, 0)),
])
def test_expand_dims(shape, axis):
  expanded_shape = np.expand_dims(np.zeros(shape), axis).shape
  _test_op(shape, npg.expand_dims, torch.reshape, 
    npg_kwargs={'axis': axis}, 
    torch_kwargs={'shape': expanded_shape}, 
  )



@pytest.mark.parametrize("shapes, axis", [
  ([(3,), (4,)], 0),
  ([(3,), (4,), (5,)], 0),
  ([(2, 3), (2, 3)], 0),
  ([(3, 4), (3, 4)], 0),
  ([(2, 3, 4), (2, 3, 5)], 2),
  ([(2, 3, 4), (2, 4, 4)], 1),
  ([(2, 3, 4), (2, 3, 5)], -1),
  ([(1, 0), (1, 0)], 0),
  ([(1, 0), (1, 0)], 1),
  ([(1,), (1,), (1,)], 0),
  ([(2, 3, 4), (2, 3, 5), (2, 3, 6)], 2),
  ([(1, 2, 3), (1, 3, 3), (1, 4, 3)], 1),
])
def test_concat(shapes, axis):
  _test_concat(shapes, axis)


@pytest.mark.parametrize("shape, index", [
  ((1,), np.s_[0]),
  ((4,), np.s_[0]),
  ((4,), np.s_[1]),
  ((4,2), np.s_[0,0]),
  ((4,3), np.s_[2,2]),
  ((4,3), np.s_[:,2]),
  ((4,3), np.s_[:,:]),
  ((4,3,5), np.s_[0,:,2]),
  ((4,3,5), np.s_[-1,:,2:4]),
  ((4,3,5), np.s_[:-1,:,:]),
  ((10,6,5), np.s_[:-1,:,:]),
  ((10,6,5), np.s_[:-1,2:4,:]),
  ((10,12,5), np.s_[:,::2,:]),
  ((10,12,5), np.s_[4,2:9:2,:]),
  ((10,12,10), np.s_[np.random.randint(0,9,(6,7)),2:9:2,:]),
  ((10,12,10), 
    np.s_[
      np.random.randint(0,9,(6,7)),
      2:9:2,
      np.random.randint(0,9,(6,7))
    ]
  ),
  ((10,12,10), 
    np.s_[
      2:9:2,
      np.random.randint(0,9,(6,7)),
      np.random.randint(0,9,(6,7))
    ]
  ),
  ((10,12,10), 
    np.s_[
      2:9:2,
      np.random.randint(0,2, (12,)).astype('bool'),
      :
    ]
  ),
  ((10,12,10), 
    np.s_[
      0:7,
      np.random.randint(0,2, (12,10)).astype('bool'),
    ],
  ),
])
def test_get(shape, index):
  _test_index(shape, index)



@pytest.mark.parametrize("shape, mask", [
  ((), 2),
  ((1,), -1.0),
  ((4,), 1.0),
  ((4,2), 7),
  ((4,3,5), 8),
  ((10,6,5), 3),
])
def test_masked_fill(shape, mask):
  _test_masked_fill_(shape, mask)



@pytest.mark.parametrize("shape1, shape2", [
  ((4,5), (5,2)),
  ((1,4,5), (3,5,2)),
  ((1,4,5), (9,12,3,5,2)),
  ((7,6,2,1,4,5), (5,2)),
  ((4,5), (5,)),
  ((9,11,7,4,5), (5,)),
  ((5,), (5,4)),
  ((5,), (8,9,11,5,4)),
])
def test_matmul(shape1, shape2):
  _test_magic_method(shape1, shape2, '__matmul__')




@pytest.mark.parametrize("shapes, subscripts", [
  (((3, 4), (4, 2)), "ij,jk->ik"),
  (((5,), (3,)), "i,j->ij"),
  (((2, 3, 4), (2, 4, 5)), "bij,bjk->bik"),
  (((3, 4),(4, 7)), "ij,jk"),
  (((3, 4, 5), (4, 3, 2)), "ijk,jil"),
  (((4, 4),), "ii->"),
  (((3, 4, 5, 4),), "ijkj->ik"),
  (((3, 4, 4, 5, 7),), "ijjkl->ik"),
  (((3, 4, 4, 8, 7, 8, 7),), "ijjklkl->ik"),
  (((3, 4, 4, 8, 7, 7, 8),), "ijjkllk->ik"),

  (((3, 4, 5, 7), (2, 4, 5, 8)), "ijkl,xjky->ik"),
  (((3, 4, 5, 7), (2, 4, 5, 8, 8)), "ijkl,xjkyy->ik"),
  (((3, 4, 5, 7), (8, 2, 4, 5, 8)), "ijkl,yxjky->ik"),
  (((3, 4, 5, 7), (2, 8, 4, 5, 8)), "ijkl,xyjky->ik"),
  (((3, 4, 5, 7), (8, 2, 4, 5, 5, 8)), "ijkl,yxjkky->ik"),
  (((3, 4, 5, 7),(2, 8, 9)), "ijkl,xyz->ik"),  

  (((2, 1, 3, 4), (1, 5, 4, 3)), "aicd,bjdc->abij"),
  (((2, 3, 4), (2, 5, 6)), "bik,bjl->bijkl"),
  (((3, 3, 3, 3),), "iiii->i"), 
  (((3, 3, 3, 6, 3),), "iiiki->i"), 
  (((4, 5, 4, 2), (4, 5, 2)), "ijik,ijk->ijk"),
  (((4, 5, 4, 2), (4, 5, 4)), "ijik,iji->ijk"),

  (((2, 3, 4), (4, 5), (5, 6)), "ijk,kl,lm->ijm"),
  (((2, 3, 4), (4, 7, 5), (5, 6, 3)), "abc,cde,efb->adf"),
])
def test_einsum(shapes, subscripts):
  _test_einsum(shapes, subscripts)


@pytest.mark.parametrize("shape, k, s, p, d", [
  # Basic cases
  ((1, 1, 4, 4), 2, 1, 0, 1),
  ((1, 3, 8, 8), 3, 1, 1, 1),
  ((2, 3, 16, 16), 2, 2, 0, 1),

  # Various input shapes
  ((1, 1, 5, 5), 3, 1, 1, 1),
  ((1, 3, 8, 8), 2, 2, 0, 1),
  ((2, 4, 10, 12), 3, 1, 1, 1),

  # Different kernel sizes
  ((1, 2, 8, 8), (3, 2), 1, (1, 0), 1),
  ((1, 2, 9, 9), 4, 1, 1, 1),
  ((1, 3, 16, 16), (5, 3), 1, (2, 1), 1),

  # Padding variations
  ((1, 1, 6, 6), 3, 1, 1, 1),
  ((1, 2, 8, 8), 5, 1, 2, 1),
  ((1, 3, 7, 7), 3, 1, (0, 1), 1),

  # Stride variations
  ((1, 1, 11, 11), 2, 3, 0, 1),
  ((1, 2, 9, 9), 3, (2, 1), 1, 1),

  # Dilation variations
  ((1, 1, 9, 9), 3, 1, 1, 2),
  ((1, 3, 13, 13), (3, 2), 1, (1, 0), (2, 1)),

  # Combination of different parameters
  ((2, 3, 15, 15), 3, 2, 1, 1),
  ((1, 4, 21, 21), (5, 3), (2, 1), (2, 1), 2),

  # Edge cases
  ((1, 1, 3, 3), 3, 1, 0, 1),
  ((1, 1, 2, 2), 2, 2, 0, 1),
  ((1, 1, 5, 5), 5, 1, 0, 1),

  # Large input sizes
  ((1, 3, 64, 64), 4, 2, 1, 1),
  ((2, 3, 129, 129), 5, 2, 2, 1),

  # More channels
  ((1, 16, 8, 8), 3, 1, 1, 1),
  ((2, 32, 16, 16), 2, 2, 0, 1),

  # 1D-like cases
  ((1, 1, 1, 10), (1, 3), 1, (0, 1), 1),
  ((1, 1, 10, 1), (3, 1), 1, (1, 0), 1),
])
def test_pooling(shape, k, s, p, d):
  _test_conv2d(shape, k, s, p, d)
  _test_pooling(shape, k, s, p, d, 'max')
  _test_pooling(shape, k, s, p, d, 'avg')




def _backward(c, ct):
  _y = np.random.randn(*ct.shape)
  y = npg.ndarray(_y)
  yt = torch.tensor(_y)

  npg.sum(c * y).backward()
  torch.sum(ct * yt).backward()

def _allclose(a, b):
  mask = ~(np.isnan(a) | np.isnan(b))
  return np.allclose(a[mask], b[mask])

def _test_op(shape, npg_op, torch_op, 
              npg_args=(), npg_kwargs={}, torch_args=(), torch_kwargs={}):
  _x = np.random.randn(*shape)

  x = npg.ndarray(_x)
  c = npg_op(x, *npg_args, **npg_kwargs)
  xt = torch.tensor(_x, requires_grad=True)
  ct = torch_op(xt, *torch_args, **torch_kwargs)

  _backward(c, ct)

  assert _allclose(c.data, ct.detach().numpy())
  assert _allclose(x.grad, xt.grad.numpy())

def _test_magic_method(shape1, shape2, magic_method):
  _a = np.random.randn(*shape1)
  _b = np.random.randn(*shape2)

  a = npg.ndarray(_a)
  b = npg.ndarray(_b)
  c = getattr(a, magic_method)(b)
  
  at = torch.tensor(_a, requires_grad=True)
  bt = torch.tensor(_b, requires_grad=True)
  ct = getattr(at, magic_method)(bt) 

  _backward(c, ct)

  assert _allclose(ct.detach().numpy(), c.data)
  assert _allclose(a.grad, at.grad.numpy())
  assert _allclose(b.grad, bt.grad.numpy())


def _test_concat(shapes, axis):
  _xs = [np.random.randn(*shape) for shape in shapes]

  xs = [npg.ndarray(_x) for _x in _xs]
  c = npg.concat(xs, axis=axis)
  xts = [torch.tensor(_x, requires_grad=True) for _x in _xs]
  ct = torch.concat(xts, dim=axis)

  _backward(c, ct)

  assert _allclose(c.data, ct.detach().numpy())
  assert all([
    _allclose(x.grad, xt.grad.numpy()) for x, xt, in zip(xs, xts)
  ])

def _test_index(shape, index):
  _x = np.random.randn(*shape)

  x = npg.ndarray(_x)
  c = x[index]
  xt = torch.tensor(_x, requires_grad=True)
  ct = xt[index]

  _backward(c, ct)

  assert _allclose(c.data, ct.detach().numpy())
  assert _allclose(x.grad, xt.grad.numpy())

def _test_masked_fill_(shape, value):
  _x = np.random.randn(*shape)
  _mask = np.random.randint(0, 2, size=shape).astype('bool')

  x = npg.ndarray(_x)
  mask = npg.ndarray(_mask)
  c = npg.masked_fill(x, mask, value)

  xt = torch.tensor(_x, requires_grad=True)
  maskt = torch.tensor(_mask, dtype=torch.bool)
  ct = xt.masked_fill(maskt, value)

  assert _allclose(ct.detach().numpy(), c.data)


def _test_einsum(shapes, subscripts):
  _xs = [np.random.randn(*shape) for shape in shapes]

  xs = [npg.ndarray(_x) for _x in _xs]
  c = npg.einsum(subscripts, *xs)
  xts = [torch.tensor(_x, requires_grad=True) for _x in _xs]
  ct = torch.einsum(subscripts, *xts)

  _backward(c, ct)

  assert _allclose(c.data, ct.detach().numpy())
  assert all([
    _allclose(x.grad, xt.grad.numpy()) for x, xt, in zip(xs, xts)
  ])

def _test_pooling(shape, k, s, p, d, pooling):
  _x = np.random.randn(*shape)

  x = npg.ndarray(_x)
  xt = torch.tensor(_x, requires_grad=True)
  if pooling == 'max':
    c = npg.max_pool2d(x, k, s, p, d)
    ct = torch.nn.functional.max_pool2d(xt, k, s, p, d)
  elif pooling == 'avg':
    c = npg.avg_pool2d(x, k, s, p)
    ct = torch.nn.functional.avg_pool2d(xt, k, s, p)

  _backward(c, ct)

  assert _allclose(c.data, ct.detach().numpy())
  assert _allclose(x.grad, xt.grad.numpy())


def _test_conv2d(shape, k, s, p, d):
  out_channel = np.random.randint(4, 15)
  in_channel = shape[1]

  kh, kw = (k, k) if isinstance(k, int) else (k[0], k[1])
  _x = np.random.randn(*shape)
  _W = np.random.randn(out_channel, in_channel, kh, kw)
  _b = np.random.randn(out_channel)

  x = npg.ndarray(_x)
  W = npg.ndarray(_W)
  b = npg.ndarray(_b)

  xt = torch.tensor(_x, requires_grad=True)
  Wt = torch.tensor(_W, requires_grad=True)
  bt = torch.tensor(_b, requires_grad=True)

  c = npg.conv2d(x, W, b, s, p, d)
  ct = torch.nn.functional.conv2d(xt, Wt, bt, s, p, d)

  _backward(c, ct)

  assert _allclose(c.data, ct.detach().numpy())
  assert _allclose(x.grad, xt.grad.numpy())
  assert _allclose(W.grad, Wt.grad.numpy())
  assert _allclose(b.grad, bt.grad.numpy())
