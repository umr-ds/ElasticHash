from random import random

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
from datetime import datetime

from PIL.Image import Image
from tensorboard.compat.proto.summary_pb2 import HistogramProto
from skimage.transform import resize as resize_array


def resize_image(image, height, width,
                 resize_mode=None,
                 ):
    """
    Modified from https://github.com/NVIDIA/DIGITS/blob/master/digits/utils/image.py

    Resizes an np.array
    Arguments:
    image -- numpy.ndarray
    height -- height of new image
    width -- width of new image
    Keyword Arguments:
    resize_mode -- can be crop, squash, fill or half_crop
    """

    channels = 3

    if resize_mode is None:
        resize_mode = 'squash'
    if resize_mode not in ['crop', 'squash', 'fill', 'half_crop', 'random_crop']:
        raise ValueError('resize_mode "%s" not supported' % resize_mode)

    # convert to array
    # image = image_to_array(image, channels)

    # No need to resize
    if image.shape[0] == height and image.shape[1] == width:
        return image

    # Resize
    width_ratio = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height
    if resize_mode == 'squash' or width_ratio == height_ratio:
        return resize_array(image, (height, width))
    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = height
            resize_width = int(round(image.shape[1] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / width_ratio))
        image = resize_array(image, (resize_height, resize_width))

        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width - width) / 2.0))
            return image[:, start:start + width]
        else:
            start = int(round((resize_height - height) / 2.0))
            return image[start:start + height, :]
    elif resize_mode == 'random_crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = height
            resize_width = int(round(image.shape[1] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / width_ratio))
        image = resize_array(image, (resize_height, resize_width))

        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width - width) / 2.0))
            start = int(min(max(np.random.normal(start, 400), 0), resize_width - width))
            return image[:, start:start + width]
        else:
            start = int(round((resize_height - height) / 2.0))
            start = int(min(max(np.random.normal(start, 400), 0), resize_width - width))
            return image[start:start + height, :]
    else:
        if resize_mode == 'fill':
            # resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio:
                resize_width = width
                resize_height = int(round(image.shape[0] / width_ratio))
                if (height - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = height
                resize_width = int(round(image.shape[1] / height_ratio))
                if (width - resize_width) % 2 == 1:
                    resize_width += 1
            image = resize_array(image, (resize_height, resize_width))
        elif resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(image.shape[1] / new_ratio))
            resize_height = int(round(image.shape[0] / new_ratio))
            if width_ratio > height_ratio and (height - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (width - resize_width) % 2 == 1:
                resize_width += 1
            image = resize_array(image, (resize_height, resize_width))
            # chop off ends of dimension that is still too long
            if width_ratio > height_ratio:
                start = int(round((resize_width - width) / 2.0))
                image = image[:, start:start + width]
            else:
                start = int(round((resize_height - height) / 2.0))
                image = image[start:start + height, :]
        else:
            raise Exception('unrecognized resize_mode "%s"' % resize_mode)

        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = (height - resize_height) / 2
            noise_size = (padding, width)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=0)
        else:
            padding = (width - resize_width) / 2
            noise_size = (height, padding)
            if channels > 1:
                noise_size += (channels,)
            noise = np.random.randint(0, 255, noise_size).astype('uint8')
            image = np.concatenate((noise, image, noise), axis=1)

        return image


def mAP(y_true, y_score, k=None):
    n_scores, n_classes = y_true.shape
    y_score_neg = -y_score
    aps = []
    k = n_scores if (k is None or k > n_scores) else k
    for c in range(n_classes):
        ids = y_score_neg[:, c].argsort()
        y_true_sorted = y_true[ids, c]
        ap = .0
        tp = .0
        for n, t in enumerate(y_true_sorted[:k], 1):
            tp += t
            ap += tp / n
        aps += [ap / n_scores]
    return np.mean(aps), aps


def create_dirs(base_dir='./'):
    """
    Create dirs for snapshots and log files
    :param base_dir: where to create dirs
    :return: path to logs, path to snapshots
    """
    timestamp = datetime.now()
    suffix = timestamp.strftime("%Y_%m_%d_%H%M%S")
    log_dir = 'logs_' + suffix
    chkpt_dir = 'checkpoints_' + suffix
    if not os.path.exists(os.path.join(base_dir, chkpt_dir)):
        os.makedirs(os.path.join(base_dir, chkpt_dir))
    if not os.path.exists(os.path.join(base_dir, log_dir)):
        os.makedirs(os.path.join(base_dir, log_dir))
    if os.path.islink(os.path.join(base_dir, "logs")):
        os.remove(os.path.join(base_dir, "logs"))
    os.symlink(log_dir, os.path.join(base_dir, "logs"))
    if os.path.islink(os.path.join(base_dir, "checkpoints")):
        os.remove(os.path.join(base_dir, "checkpoints"))
    os.symlink(chkpt_dir, os.path.join(base_dir, "checkpoints"))
    return log_dir, chkpt_dir


def one_hot(ids, size):
    v = np.zeros(size)
    v[ids] = 1
    return v


def euclidean_distance_np(a, b):
    diffs = a[:, np.newaxis] - b
    diffs = np.square(diffs)
    dists = np.sqrt(np.sum(diffs, axis=2))
    return dists


def log_histogram(writer, tag, values, step, bins=1000):
    values = np.array(values)
    counts, bin_edges = np.histogram(values, bins=bins)
    hist = HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    bin_edges = bin_edges[1:]

    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    tf.summary.histogram(
        tag,
        hist,
        step=step,
        buckets=bins,
        description=None
    )


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def random_erasing(image, area=0.6, prob=0.6):
    if (random.random() < prob):
        w, h = image.size

        w_occlusion_max = int(w * area)
        h_occlusion_max = int(h * area)

        w_occlusion_min = int(w * 0.1)
        h_occlusion_min = int(h * 0.1)

        w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
        h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

        if len(image.getbands()) == 1:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
        else:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, len(image.getbands())) * 255))

        random_position_x = random.randint(0, w - w_occlusion)
        random_position_y = random.randint(0, h - h_occlusion)

        image.paste(rectangle, (random_position_x, random_position_y))

    return image


def euclidean_distance_np(a, b):
    diffs = a[:, np.newaxis] - b
    diffs = np.square(diffs)
    dists = np.sqrt(np.sum(diffs, axis=2))
    return dists


def log_histogram(writer, tag, values, step, bins=1000):
    values = np.array(values)
    counts, bin_edges = np.histogram(values, bins=bins)
    hist = HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    bin_edges = bin_edges[1:]

    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    tf.summary.histogram(
        tag,
        hist,
        step=step,
        buckets=bins,
        description=None
    )


# Embedding visualization
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
    data: NxHxW[x3] tensor containing the images.
    Returns:
    data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min_value = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose((1, 2, 3, 0)) - min_value).transpose((3, 0, 1, 2))
    max_value = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose((1, 2, 3, 0)) / max_value).transpose((3, 0, 1, 2))

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def save_labels_tsv(classes, labels, filepath, log_dir):
    with open(os.path.join(log_dir, filepath), 'w') as f:
        f.write('{}\t{}\n'.format("Bag-Id", "Name"))
        for c, label in zip(classes, labels):
            f.write('{}\t{}\n'.format(c, label))


def create_dirs(base_dir='/data/reidentification/'):
    timestamp = datetime.now()
    suffix = timestamp.strftime("%Y_%m_%d_%H%M%S")
    log_dir = 'logs_' + suffix
    chkpt_dir = 'checkpoints_' + suffix
    if not os.path.exists(os.path.join(base_dir, chkpt_dir)):
        os.makedirs(os.path.join(base_dir, chkpt_dir))
    if not os.path.exists(os.path.join(base_dir, log_dir)):
        os.makedirs(os.path.join(base_dir, log_dir))
    if os.path.islink(os.path.join(base_dir, "logs")):
        os.remove(os.path.join(base_dir, "logs"))
    os.symlink(log_dir, os.path.join(base_dir, "logs"))
    if os.path.islink(os.path.join(base_dir, "checkpoints")):
        os.remove(os.path.join(base_dir, "checkpoints"))
    os.symlink(chkpt_dir, os.path.join(base_dir, "checkpoints"))
    return log_dir, chkpt_dir
