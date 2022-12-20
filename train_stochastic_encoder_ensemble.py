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
from networks import NatureDQNNetwork2, NatureDQNEncoder, Ensemble, TanhNormal
import matplotlib.pyplot as plt

import tensorflow_probability
from jaxrl5.distributions.tanh_transformed import TanhTransformedDistribution



tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


def get_datasets(rotations):
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.uint8(train_ds['image'])
    test_ds['image'] = jnp.uint8(test_ds['image'])


    filtered_indices = np.array([])
    for digit in range(10):
        filtered_indices = np.concatenate((np.where(test_ds['label']==digit)[0][:900], filtered_indices))


    for rotation in rotations:
        test_ds_rotated = []
        for img in test_ds['image']:
            rotated_img = imutils.rotate(np.array(img), angle=rotation)
            test_ds_rotated.append(np.expand_dims(rotated_img, axis=-1))
        test_ds_rotated = np.array(test_ds_rotated)
        test_ds["image"+str(rotation)] = np.array(test_ds_rotated[(filtered_indices).astype(int)])
    test_ds["label"] = test_ds["label"][(filtered_indices).astype(int)]

    return train_ds, test_ds



@jax.jit
def update_model(rng, state, images, labels, kl_coeff):
    key, rng = jax.random.split(rng)
    images = images.astype(jnp.float32) / 255.0
    encoder_state, final_layers_state = state
    
    def loss_fn(encoder_params, final_layers_params, images, labels):
        z_dist = encoder_state.apply_fn(
            {
                'params': encoder_params,
            },
            images)

        z = z_dist.sample(seed=key)


        qs = final_layers_state.apply_fn(
            {
                'params': final_layers_params,
            },
            z)

        batch_size = len(labels)

        # # regressing to label - action space 10
        # one_hot = jax.nn.one_hot(labels, 10)
        # one_hot = jnp.tile(one_hot, (len(qs), 1, 1))
        # loss = jnp.mean((qs-one_hot)**2)


        # mnist driving - action space 50 - reshape to 10 x 5
        discretized_a1s = np.expand_dims(np.expand_dims(np.array([_ for _ in range(10)]), axis=1), axis=0)*np.ones([batch_size, 10, 5])
        discretized_a2s = np.expand_dims(np.expand_dims(np.array([0.2*_ for _ in range(5)]), axis=0), axis=0) *np.ones([batch_size, 10, 5])
        labels = jnp.expand_dims(jnp.expand_dims(labels, axis=1), axis=2)*jnp.ones([batch_size, 10, 5])
        returns = discretized_a2s - jnp.abs(discretized_a1s-labels)*(discretized_a2s**2)*2
        # returns = discretized_a2s - jnp.abs(discretized_a1s-labels)*jnp.sqrt(discretized_a2s)
        returns =jnp.expand_dims(returns, 0)
        loss = jnp.mean(jnp.abs(jnp.reshape(qs, (-1, batch_size, 10, 5)) - returns))

        prior_dist = tfd.MultivariateNormalDiag(loc=jnp.zeros(z.shape),
                                                  scale_diag=1*jnp.ones(z.shape))
        loss += kl_coeff*tfd.kl_divergence(
                z_dist, prior_dist, allow_nan_stats=True, name=None
            ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False, argnums=(0, 1))
    loss, grads = grad_fn(encoder_state.params, final_layers_state.params, images, labels)
    encoder_state = encoder_state.apply_gradients(grads=grads[0])
    final_layers_state = final_layers_state.apply_gradients(grads=grads[1])

    state = (encoder_state, final_layers_state)
    return rng, state, loss

@jax.jit
def evaluate_model(state, images, labels):
    images = images.astype(jnp.float32) / 255.0
    encoder_state, final_layers_state = state
    z_dist = encoder_state.apply_fn(
        {
            'params': encoder_state.params,
        },
        images)

    z = z_dist._loc
    qs = final_layers_state.apply_fn(
        {
            'params': final_layers_state.params,
        },
        z)

    batch_size = len(labels)

    discretized_a1s = np.expand_dims(np.expand_dims(np.array([_ for _ in range(10)]), axis=1), axis=0)*np.ones([batch_size, 10, 5])
    discretized_a2s = np.expand_dims(np.expand_dims(np.array([0.2*_ for _ in range(5)]), axis=0), axis=0) *np.ones([batch_size, 10, 5])
    labels = jnp.expand_dims(jnp.expand_dims(labels, axis=1), axis=2)*jnp.ones([batch_size, 10, 5])

    # returns = discretized_a2s - jnp.abs(discretized_a1s-labels)*jnp.sqrt(discretized_a2s)
    # returns = discretized_a2s - jnp.abs(discretized_a1s-labels)*(discretized_a2s)#*2
    returns = discretized_a2s - jnp.abs(discretized_a1s-labels)*(discretized_a2s**2)*2
    returns =jnp.expand_dims(returns, 0)
    # loss = jnp.mean((jnp.reshape(qs, (-1, batch_size, 10, 5)) - returns)**2)
    loss = jnp.mean(jnp.abs(jnp.reshape(qs, (-1, batch_size, 10, 5)) - returns))

    best_actions = jnp.argmax(jnp.mean(qs, axis=0), axis=1)
    flattened_returns = np.reshape(returns, (batch_size, 50))
    policy_avg_return = jnp.mean(jnp.take_along_axis(flattened_returns, jnp.expand_dims(best_actions, axis=-1), axis=-1).squeeze())

    best_robust_actions = jnp.argmax(jnp.min(qs, axis=0), axis=1)
    robust_policy_avg_return = jnp.mean(jnp.take_along_axis(flattened_returns, jnp.expand_dims(best_robust_actions, axis=-1), axis=-1).squeeze())

    return loss, policy_avg_return, robust_policy_avg_return, jnp.mean(jnp.std(qs, axis=0))

# @jax.jit
# def get_model_outputs(state, images, labels):
#     images = images.astype(jnp.float32) / 255.0
#     encoder_state, final_layers_state = state
#     z = encoder_state.apply_fn(
#         {
#             'params': encoder_state.params,
#         },
#         images)
#     qs = final_layers_state.apply_fn(
#         {
#             'params': final_layers_state.params,
#         },
#         z)
#     return qs, z

def create_state(learning_rate, rng):


    encoder_cls = partial(NatureDQNEncoder)
    # encoder = Ensemble(encoder_cls, num=5)

    latent_dim = 4
    encoder_cls = partial(Ensemble, net_cls=encoder_cls, num=20)
    encoder = TanhNormal(encoder_cls, output_dim=latent_dim)
    encoder_variables = encoder.init(rng, jnp.ones([1, 28, 28, 1]))


    encoder_params = encoder_variables['params']
    encoder_tx = optax.adam(learning_rate)
    encoder_state = TrainState.create(apply_fn=encoder.apply,
                             params=encoder_params,
                             tx=encoder_tx)

    final_layers = NatureDQNNetwork2(action_dim=50)
    final_layers_variables = final_layers.init(rng, jnp.ones([5, 1, 4]))
    final_layers_params = final_layers_variables['params']
    final_layers_tx = optax.adam(learning_rate)
    final_layers_state = TrainState.create(apply_fn=final_layers.apply,
                             params=final_layers_params,
                             tx=final_layers_tx)
    return (encoder_state, final_layers_state)

def train_epoch(state, train_ds, batch_size, rng, epoch, kl_coeff):
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
                                             batch_labels, kl_coeff)
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
    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 20
    kl_coeff = 0.1

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    state = create_state(learning_rate, init_rng)  

    for epoch in range(num_epochs):
        rng, input_rng = jax.random.split(rng)
        state = train_epoch(state, train_ds, batch_size, input_rng, epoch, kl_coeff)
        for rotation in rotations:
            test_loss, policy_avg_return, robust_policy_avg_return, q_std = evaluate_model(state, test_ds['image'+str(rotation)],
                                                  test_ds['label'])
            print('%d loss : %.4f, return: %.4f, robust return: %.4f, q std: %.4f' % (rotation, test_loss, policy_avg_return, robust_policy_avg_return, q_std))
    return state

if __name__ == "__main__":
    rotations = np.array([0, 15, 30, 45, 60, 75])
    train_ds, test_ds = get_datasets(rotations)
    state = train(train_ds, test_ds, rotations)
    import IPython; IPython.embed()