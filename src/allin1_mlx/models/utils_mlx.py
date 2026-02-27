import mlx.core as mx
import mlx.nn as nn


class SoftmaxAxis(nn.Module):
  def __init__(self, axis: int = -1):
    super().__init__()
    self.axis = axis

  def __call__(self, x: mx.array) -> mx.array:
    return mx.softmax(x, axis=self.axis)


class LogSoftmaxAxis(nn.Module):
  def __init__(self, axis: int = -1):
    super().__init__()
    self.axis = axis

  def __call__(self, x: mx.array) -> mx.array:
    return mx.log_softmax(x, axis=self.axis)


def get_activation_function(name: str):
  activation_functions = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'softmax': SoftmaxAxis(axis=1),
    'log_softmax': LogSoftmaxAxis(axis=1),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'gelu': nn.GELU(),
    'prelu': nn.PReLU(),
  }

  if name in activation_functions:
    return activation_functions[name]
  raise ValueError(f"Unsupported activation function: {name}")
