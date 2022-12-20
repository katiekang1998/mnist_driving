import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training.train_state import TrainState
from functools import partial
import imutils

tf.config.experimental.set_visible_devices([], "GPU")
from typing import Any

import jax.numpy as jnp
import tqdm
from networks import NatureDQNNetwork2, NatureDQNEncoder, Q, Ensemble2
import matplotlib.pyplot as plt


def get_datasets(rotations):
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.uint8(train_ds['image'])
    test_ds['image'] = jnp.uint8(test_ds['image'])

    for rotation in rotations:
        test_ds_rotated = []
        for img in test_ds['image']:
            rotated_img = imutils.rotate(np.array(img), angle=rotation)
            test_ds_rotated.append(np.expand_dims(rotated_img, axis=-1))
        test_ds["image"+str(rotation)] = np.array(test_ds_rotated)

    return train_ds, test_ds



@jax.jit
def update_model(rng, state, images, labels):
    key, rng = jax.random.split(rng)
    images = images.astype(jnp.float32) / 255.0
    
    def loss_fn(params, images, labels):
        qs, z = state.apply_fn(
            {
                'params': params,
            },
            images)

        batch_size = len(labels)

        # # regressing to label - action space 10
        # one_hot = jax.nn.one_hot(labels, 10)
        # one_hot = jnp.tile(one_hot, (len(qs), 1, 1))
        # loss = jnp.mean((qs-one_hot)**2)


        # mnist driving - action space 50 - reshape to 10 x 5
        discretized_a1s = np.expand_dims(np.expand_dims(np.array([_ for _ in range(10)]), axis=1), axis=0)*np.ones([batch_size, 10, 5])
        discretized_a2s = np.expand_dims(np.expand_dims(np.array([0.2*_ for _ in range(5)]), axis=0), axis=0) *np.ones([batch_size, 10, 5])
        labels = jnp.expand_dims(jnp.expand_dims(labels, axis=1), axis=2)*jnp.ones([batch_size, 10, 5])
        returns = discretized_a2s - jnp.abs(discretized_a1s-labels)*(discretized_a2s**2)
        returns =jnp.expand_dims(returns, 0)
        loss = jnp.mean((jnp.reshape(qs, (-1, batch_size, 10, 5)) - returns)**2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params, images, labels)
    state = state.apply_gradients(grads=grads)
    return rng, state, loss

@jax.jit
def evaluate_model(state, images, labels):
    images = images.astype(jnp.float32) / 255.0
    qs, z = state.apply_fn(
        {
            'params': state.params,
        },
        images)

    batch_size = len(labels)

    discretized_a1s = np.expand_dims(np.expand_dims(np.array([_ for _ in range(10)]), axis=1), axis=0)*np.ones([batch_size, 10, 5])
    discretized_a2s = np.expand_dims(np.expand_dims(np.array([0.2*_ for _ in range(5)]), axis=0), axis=0) *np.ones([batch_size, 10, 5])
    labels = jnp.expand_dims(jnp.expand_dims(labels, axis=1), axis=2)*jnp.ones([batch_size, 10, 5])

    returns = discretized_a2s - jnp.abs(discretized_a1s-labels)*(discretized_a2s**2)
    returns =jnp.expand_dims(returns, 0)
    loss = jnp.mean((jnp.reshape(qs, (-1, batch_size, 10, 5)) - returns)**2)

    best_actions = jnp.argmax(jnp.mean(qs, axis=0), axis=1)
    flattened_returns = np.reshape(returns, (batch_size, 50))
    policy_avg_return = jnp.mean(jnp.take_along_axis(flattened_returns, jnp.expand_dims(best_actions, axis=-1), axis=-1).squeeze())

    best_robust_actions = jnp.argmax(jnp.min(qs, axis=0), axis=1)
    robust_policy_avg_return = jnp.mean(jnp.take_along_axis(flattened_returns, jnp.expand_dims(best_robust_actions, axis=-1), axis=-1).squeeze())

    return loss, policy_avg_return, robust_policy_avg_return

@jax.jit
def get_model_outputs(state, images, labels):
    images = images.astype(jnp.float32) / 255.0
    qs, z = state.apply_fn(
        {
            'params': state.params,
        },
        images)
    return qs, z

def create_state(learning_rate, momentum, rng):
    encoder_cls = partial(NatureDQNEncoder)
    network_cls = partial(NatureDQNNetwork2, action_dim=50)
    # model = Q(encoder_cls, network_cls)
    model = Ensemble2(encoder_cls, network_cls, num=20)

    variables = model.init(rng, jnp.ones([1, 28, 28, 1]))
    params = variables['params']
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(apply_fn=model.apply,
                             params=params,
                             tx=tx)

def train_epoch(state, train_ds, batch_size, rng, epoch):
    train_ds_size = len(train_ds['image'])
    steps_per_epochs = train_ds_size // batch_size
    permutations = jax.random.permutation(rng, train_ds_size)
    permutations = permutations[:steps_per_epochs * batch_size]
    permutations = permutations.reshape(steps_per_epochs, batch_size)

    batch_metrics = []
    for prm in tqdm.tqdm(permutations):
        batch_images = train_ds['image'][prm]
        batch_labels = train_ds['label'][prm]
        rng, state, loss = update_model(rng, state, batch_images,
                                             batch_labels)
        metrics = {
            'loss': loss,
        }
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)

    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    print('train epoch: %d, loss: %.4f' %
          (epoch + 1, epoch_metrics_np['loss']))

    return state

def train(train_ds, test_ds, rotations):
    learning_rate = 0.1
    momentum = 0.9
    batch_size = 128
    num_epochs = 20

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    learning_rate_fn = optax.piecewise_constant_schedule(learning_rate, {
        32000: 1e-1,
        48000: 1e-1
    })
    state = create_state(learning_rate_fn, momentum, init_rng)  

    for epoch in range(num_epochs):
        rng, input_rng = jax.random.split(rng)
        state = train_epoch(state, train_ds, batch_size, input_rng, epoch)
        for rotation in rotations:
            test_loss, policy_avg_return, robust_policy_avg_return = evaluate_model(state, test_ds['image'+str(rotation)],
                                                  test_ds['label'])
            print('%d loss : %.4f, return: %.4f, robust return: %.4f' % (rotation, test_loss, policy_avg_return, robust_policy_avg_return))
    return state

if __name__ == "__main__":
    rotations = np.array([0, 15, 30, 45, 60, 75])
    train_ds, test_ds = get_datasets(rotations)
    state = train(train_ds, test_ds, rotations)
    import IPython; IPython.embed()