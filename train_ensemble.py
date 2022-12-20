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
from networks import NatureDQNNetwork, Ensemble, Ensemble2, NatureDQNNetwork2, NatureDQNEncoder
import matplotlib.pyplot as plt

# class TrainState(train_state.TrainState):
#     batch_stats: Any

def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2, ), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1, ), dtype=jnp.int32)])
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)),
                         mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def random_flip(key, img):
    do_flip = jax.random.choice(key, a=jnp.asarray([False, True]))
    return jax.lax.cond(do_flip, lambda x: jnp.fliplr(x), lambda x: x, img)


def batched_augemntation(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    imgs = jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_flip, (0, 0))(keys, imgs)


def get_datasets():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.uint8(train_ds['image'])
    test_ds['image'] = jnp.uint8(test_ds['image'])
    return train_ds, test_ds

def loss_fn(params, images, labels):
    logits, representations = state.apply_fn(
        {
            'params': params,
        },
        images)


    one_hot = jax.nn.one_hot(labels, 10)
    one_hot = jnp.tile(one_hot, (len(logits), 1, 1))
    # import IPython; IPython.embed()
    # loss = jnp.mean(
    #     optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    loss = jnp.mean((logits-one_hot)**2)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    ensemble_std = jnp.mean(jnp.std(logits, axis=0), axis=1) #jnp.mean(jnp.std(logits, axis=0))

    #mean over ensemble, get standard deviate over batch
    # normalization_over_batch = jnp.expand_dims(jnp.expand_dims(jnp.std(jnp.mean(representations, axis=0), axis=0), axis=0), axis=0)
    representation_std = representations #jnp.mean(jnp.std(representations, axis=0), axis=0)#jnp.mean(jnp.std(representations/normalization_over_batch, axis=0))
    return loss, (accuracy, ensemble_std, representation_std)
info_labels = ['accuracy', 'ensemble_std', 'representation_std']


@jax.jit
def update_model(rng, state, images, labels):
    key, rng = jax.random.split(rng)

    # images = batched_augemntation(key, images)
    images = images.astype(jnp.float32) / 255.0
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, info), grads = grad_fn(state.params, images, labels)
    state = state.apply_gradients(grads=grads)
    return rng, state, loss, info


@jax.jit
def evaluate_model(state, images, labels):
    images = images.astype(jnp.float32) / 255.0

    # logits = state.apply_fn(
    #     {
    #         'params': state.params,        },
    #     images,)
    # one_hot = jax.nn.one_hot(labels, 10)
    # loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    # accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    loss, info = loss_fn(state.params, images, labels)
    return loss, info


def create_state(learning_rate, momentum, rng):

    # model_cls = partial(NatureDQNNetwork, action_dim=10)
    # model = Ensemble(model_cls, num=5)

    model_cls = partial(NatureDQNEncoder)
    model_cls2 = partial(NatureDQNNetwork2, action_dim=10)
    model = Ensemble2(model_cls, model_cls2, num=5)


    # model = NatureDQNNetwork(action_dim=10)
    variables = model.init(rng, jnp.ones([1, 28, 28, 1]))
    params = variables['params']
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(apply_fn=model.apply,
                             params=params,
                             tx=tx)


# def compute_metrics(logits, labels):
#     one_hot_labels = jax.nn.one_hot(labels, 10)
#     loss = jnp.mean(
#         optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels))
#     accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
#     metrics = {
#         'loss': loss,
#         'accuracy': accuracy,
#     }
#     return metrics


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
        rng, state, loss, info = update_model(rng, state, batch_images,
                                             batch_labels)
        metrics = {
            'loss': loss,
            # 'accuracy': acc,
            # 'ensemble_std': ensemble_std,
        }
        for j in range(len(info)):
            metrics[info_labels[j]] = info[j]
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)

    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    print(epoch_metrics_np)
    # print('train epoch: %d, loss: %.4f, accuracy: %.2f, ensemble_std: %.4f' %
    #       (epoch + 1, epoch_metrics_np['loss'],
    #        epoch_metrics_np['accuracy'] * 100, epoch_metrics_np['ensemble_std']))

    return state


learning_rate = 0.1
momentum = 0.9
batch_size = 128
# num_updates = 60000
num_epochs = 10

train_ds, test_ds = get_datasets()
test_ds_rotated_list = []
rotations = np.array([15, 30, 45, 60, 75, 90])
for rotation in rotations:
    test_ds_rotated = []
    for img in test_ds['image']:
        rotated_img = imutils.rotate(np.array(img), angle=rotation)
        test_ds_rotated.append(np.expand_dims(rotated_img, axis=-1))
    test_ds_rotated_list.append(jnp.uint8(test_ds_rotated))

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

learning_rate_fn = optax.piecewise_constant_schedule(learning_rate, {
    32000: 1e-1,
    48000: 1e-1
})
state = create_state(learning_rate_fn, momentum, init_rng)

# num_epochs = round(0.5 + num_updates / (len(train_ds['image']) / batch_size))

for epoch in range(num_epochs):
    rng, input_rng = jax.random.split(rng)
    state = train_epoch(state, train_ds, batch_size, input_rng, epoch)
    test_loss, info = evaluate_model(state, test_ds['image'],
                                              test_ds['label'])
    # print(' test epoch: %d, loss: %.2f, accuracy: %.2f, ensemble_std: %.4f' %
    #       (epoch + 1, test_loss, test_accuracy * 100, ensemble_std))

    representation_stds = []
    metrics = {'loss': test_loss.item()}
    for j in range(len(info)):
        metrics[info_labels[j]] = info[j]#.item()
    print(metrics)
    representation_stds.append(metrics["representation_std"])



    for i in range(len(rotations)):
        print(rotations[i])
        test_loss, info = evaluate_model(state, test_ds_rotated_list[i],
                                              test_ds['label'])
        # print(' test epoch: %d, loss: %.2f, accuracy: %.2f, ensemble_std: %.4f' %
        #       (epoch + 1, test_loss, test_accuracy * 100, ensemble_std))

        metrics = {'loss': test_loss.item()}
        for j in range(len(info)):
            metrics[info_labels[j]] = info[j]#.item()
        print(metrics)
        representation_stds.append(metrics["representation_std"])

    # import IPython; IPython.embed()
    if epoch == num_epochs-1:
        fig, axs = plt.subplots(len(rotations)+1, 1, sharex=True, sharey=True, figsize=(12,10*(len(rotations)+1)))
        for k in range(len(rotations)+1):
            # import IPython; IPython.embed()
            # axs[k].hist(representation_stds[k], bins=10)
            # axs[k].axvline(jnp.mean(representation_stds[k]), c = "red")


            normalization = jnp.clip(jnp.expand_dims(jnp.std(representation_stds[0],axis=1), axis=1), a_min=0.00000000001)
            normalized_representation_stds = jnp.std(representation_stds[k]/normalization, axis=0)#jnp.mean(jnp.std(representation_stds[k]/normalization, axis=0), axis=0)
            # normalized_representation_stds = jnp.clip(normalized_representation_stds, a_max = jnp.percentile(normalized_representation_stds, 99))
            plot_dimension = 0
            axs[k].hist(normalized_representation_stds[:, plot_dimension], bins=20)
            axs[k].axvline(jnp.mean(normalized_representation_stds[:, plot_dimension]), c = "red")
            if k == 0:
                title = "0"
            else:
                title = rotations[k-1]
            axs[k].set_title(title)

        # plt.savefig("ensemble_independent.jpg")
        plt.show()

        import IPython; IPython.embed()

        fig, axs = plt.subplots(len(rotations)+1, 1, sharex=True, sharey=True, figsize=(12,10*(len(rotations)+1)))
        for k in range(len(rotations)+1):
            # import IPython; IPython.embed()
            # axs[k].hist(representation_stds[k], bins=10)
            # axs[k].axvline(jnp.mean(representation_stds[k]), c = "red")


            normalization = jnp.clip(jnp.expand_dims(jnp.std(representation_stds[0],axis=1), axis=1), a_min=0.00000000001)
            normalized_representation_stds = jnp.std(representation_stds[k]/normalization, axis=0)#jnp.mean(jnp.std(representation_stds[k]/normalization, axis=0), axis=0)
            # normalized_representation_stds = jnp.clip(normalized_representation_stds, a_max = jnp.percentile(normalized_representation_stds, 99))
            axs[k].hist(jnp.mean(normalized_representation_stds, axis=1), bins=20)
            axs[k].axvline(jnp.mean(normalized_representation_stds), c = "red")
            if k == 0:
                title = "0"
            else:
                title = rotations[k-1]
            axs[k].set_title(title)

        # plt.savefig("ensemble_independent.jpg")
        plt.show()


