from typing import Type

import flax.linen as nn
import jax
import jax.numpy as jnp

from functools import partial
from typing import Any, Callable, Sequence, Tuple
from flax import linen as nn

from jaxrl5.networks.common import default_init
from jaxrl5.distributions.tanh_transformed import TanhTransformedDistribution

class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)






class NatureDQNNetwork(nn.Module):
  action_dim: int

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

    x = nn.Conv(32,
            kernel_size=(8, 8),
            strides=(4, 4),
            kernel_init=default_init(),
            padding='SAME')(x)
    x = nn.relu(x)

    x = nn.Conv(64,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=default_init(),
            padding='SAME')(x)
    x = nn.relu(x)

    x = nn.Conv(64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=default_init(),
            padding='SAME')(x)
    x = nn.relu(x)

    x = x.reshape((*x.shape[:-3], -1))

    x = nn.Dense(64, kernel_init=default_init())(x)

    x = nn.relu(x)

    x = nn.Dense(1, kernel_init=default_init())(x)

    y = nn.Dense(512, kernel_init=default_init())(x)
    y = nn.relu(y)

    y = nn.Dense(self.action_dim, kernel_init=default_init())(y)

    return y, x






class Ensemble2(nn.Module):
    net_cls: Type[nn.Module]
    net_cls2: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        x = ensemble()(*args)
        # y = nn.vmap(self.net_cls2,
        #     in_axes=0, out_axes=0,
        #     variable_axes={'params': None},
        #     split_rngs={'params': False})()(x)

        y = self.net_cls2()(x)

        return y, x


class Q(nn.Module):
    encoder_cls: Type[nn.Module]
    network_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, o):
        z = self.encoder_cls()(o)
        qs = self.network_cls()(z)
        return qs, z

class NatureDQNEncoder(nn.Module):

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

    x = nn.Conv(32,
            kernel_size=(8, 8),
            strides=(4, 4),
            kernel_init=default_init(),
            padding='SAME')(x)
    x = nn.relu(x)

    x = nn.Conv(64,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=default_init(),
            padding='SAME')(x)
    x = nn.relu(x)

    x = nn.Conv(64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=default_init(),
            padding='SAME')(x)
    x = nn.relu(x)

    x = x.reshape((*x.shape[:-3], -1))

    x = nn.Dense(128, kernel_init=default_init())(x)

    x = nn.relu(x)
    # x = nn.Dense(4, kernel_init=default_init())(x)

    # x = nn.LayerNorm()(x)
    # x = nn.tanh(x)

    return x

class NatureDQNNetwork2(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        y = nn.Dense(512, kernel_init=default_init())(x)
        y = nn.relu(y)

        y = nn.Dense(self.action_dim, kernel_init=default_init())(y)

        return y


import functools
from typing import Optional, Type

import tensorflow_probability


tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions



class TanhNormal(nn.Module):
    base_cls: Type[nn.Module]
    output_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(self.output_dim,
                         kernel_init=default_init(),
                         name='OutputDenseMean')(x)
        log_stds = nn.Dense(self.output_dim,
                            kernel_init=default_init(),
                            name='OutputDenseLogStd')(x)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = tfd.MultivariateNormalDiag(loc=means,
                                                  scale_diag=jnp.exp(log_stds))        

        return distribution #TanhTransformedDistribution(distribution)

