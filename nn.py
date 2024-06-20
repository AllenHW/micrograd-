import random
from scalar import Scalar

class Module:
    def zero_grad(self):
      for p in self.parameters():
        p.grad = 0

    def parameters(self):
      return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Scalar(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Scalar(0)
        self.nonlin= nonlin

    def __call__(self, x):
        out = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return out.tanh() if self.nonlin else out

    def parameters(self):
      return self.w + [self.b]

    def __repr__(self):
        return ''

class Layer(Module):
    def __init__(self, nin, nout, nonlin=True):
      self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
      outs = [neuron(x) for neuron in self.neurons]
      return outs[0] if len(outs) == 1 else outs

    def parameters(self):
      return sum([n.parameters() for n in self.neurons], [])

    def __repr__(self):
      return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts):
      sz = [nin] + nouts
      self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
      for layer in self.layers:
        x = layer(x)
      return x

    def parameters(self):
      return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"