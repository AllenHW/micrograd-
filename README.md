## Npgrad

A small autograd library built with just Numpy. Inspired by [Micrograd](https://github.com/karpathy/micrograd).


### Numpy Like API

Npgrad supports subset of Numpy APIs that are commonly used in deep learning.

Most notably, Npgrad supports basic operations like `+`, `-`, `*`, `/`, `**`, `@`, `max`, `min`, `sum`, `mean`, `reshape`, `transpose`, `swapaxes`, `ravel`, `squeeze`, `expand_dims`, `concat`, `index`, `masked_fill`, `repeat`, `tile`, `einsum`, `matmul`, `max_pool2d`, `avg_pool2d`, `conv2d`.

See [APIs](https://github.com/AllenHW/npgrad/blob/main/npgrad/api.py) for a ful list of supported operations.

```python
import npgrad as npg
import numpy as np
import torch

x = npg.randn(7, 3, 4)
y = npg.randn(1, 4, 5)
z = x @ y
print(z)

# Compute gradients with torch
torch_x = torch.tensor(x.data, requires_grad=True)
torch_y = torch.tensor(y.data, requires_grad=True)
torch_z = torch_x @ torch_y

# Compare gradients
assert np.allclose(z.data, torch_z.detach().numpy())
```


### Autograd

Npgrad supports autograd. All the backward operations are implemented in Numpy. It handles broadcasting the same way as Numpy and PyTorch.

It does autograd for operations like  `advanced indexing`, `einsum`, `matmul`, `max_pool2d`, `avg_pool2d`, `conv2d`, with all the edge cases supported by Numpy.


#### Einsum

```python
x = npg.randn(3, 4, 7, 8, 3)
y = npg.randn(3, 4, 5, 6)

z = npg.sum(
  npg.einsum('ijkli,ijmn->klmn', x, y)
)
z.backward()

# Compute gradients with torch
torch_x = torch.tensor(x.data, requires_grad=True)
torch_y = torch.tensor(y.data, requires_grad=True)
torch_z = torch.sum(
  torch.einsum('ijkli,ijmn->klmn', torch_x, torch_y)
)
torch_z.backward()

# Compare gradients
assert np.allclose(x.grad, torch_x.grad.numpy())
```


#### Advanced Indexing

```python
x = npg.randn(10, 8, 12)
indices1 = np.random.randint(0, 10, (3, 2))
indices2 = np.random.randint(0, 8, (3, 2))

y = x[indices1, indices2, 2:8:2]
y = npg.sum(y)
y.backward()

# Compute gradients with torch
torch_x = torch.tensor(x.data, requires_grad=True)
torch_y = torch_x[indices1, indices2, 2:8:2]
torch_y = torch.sum(torch_y)
torch_y.backward()

# Compare gradients
assert np.allclose(x.grad, torch_x.grad.numpy())
```

And other operations are tested with PyTorch [github.com/AllenHW/npgrad/blob/main/tests/ops.py](https://github.com/AllenHW/npgrad/blob/main/tests/ops.py).

### Todo

- [ ] Support comparison operators like `==`, `!=`, `<`, `<=`, `>`, `>=`
- [ ] Support inter-operation with Python and Numpy objects for `+`, `-`, `*`, `/`, `**`, `@`
- [ ] Add checks so backward() is only called for scalars