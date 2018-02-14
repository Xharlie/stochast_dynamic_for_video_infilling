import tensorflow as tf
import random
import numpy as np


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.nn.tanh(x)


def batch_norm(inputs, name, train=True, reuse=False):
    return tf.contrib.layers.batch_norm(inputs=inputs, is_training=train,
                                        reuse=reuse, scope=name, scale=True)


def conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", reuse=False, padding='SAME'):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", reuse=False, with_w=False, padding='SAME'):
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1],
                                  input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())

        try:
            deconv = tf.nn.conv2d_transpose(input_, w,
                                            output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1],
                                            padding=padding)

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), [-1] + deconv.get_shape().as_list()[1:])

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def linear(input_, output_size, name, stddev=0.02, bias_start=0.0,
           reuse=False, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name, reuse=reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)
    idx_list_alter = np.concatenate((idx_list[minibatch_size:],idx_list[:minibatch_size]))

    minibatches = []
    minibatch_start = 0
    alter_minibatches = []
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        alter_minibatches.append(idx_list_alter[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
        alter_minibatches.append(idx_list_alter[minibatch_start:])

    return zip(range(len(minibatches)), minibatches, alter_minibatches)


def load_triplet_data_from_list(train_vids, batchidx,alter_batchidx, image_size_h, image_size_w, nShare, maxStep, flipable=True,
                                channel=1):
    seq = np.zeros((len(batchidx) , image_size_h, image_size_w, nShare * channel), dtype="float32")
    seq_same = np.zeros((len(batchidx) , image_size_h, image_size_w, nShare * channel), dtype="float32")
    seq_diff = np.zeros((len(batchidx) , image_size_h, image_size_w, nShare * channel), dtype="float32")
    for i in range(len(batchidx)):
        stidx = 0
        stidx_diff = 0
        offset_start = 0
        selected = train_vids[batchidx[i]]
        diff_selected = train_vids[alter_batchidx[i]]
        while True:
            low = 0
            high = selected.shape[2] - maxStep
            if low == high:
                stidx = 0
            elif high < 0:
                raise Exception('high < 0:len=' + str(selected.shape[2]) + ",maxStep=" + str(maxStep))
                continue
            else:
                stidx = np.random.randint(low=low, high=high)
            low = 0
            high = diff_selected.shape[2] - nShare
            if low == high:
                stidx_diff = 0
            elif high < 0:
                raise Exception('high < 0:len=' + str(diff_selected.shape[2]))
                continue
            else:
                stidx_diff = np.random.randint(low=0, high=high)
            break
        if i < len(batchidx):
            offset_start = np.random.randint(low=min(stidx + nShare, stidx + maxStep - nShare),
                                             high=stidx + maxStep - nShare)
            for j in range(nShare):
                seq[i, :, :, j:j + channel] = transform(selected[:, :, stidx + j, :])
                seq_same[i, :, :, j:j + channel] = transform(selected[:, :, offset_start + j, :])
                seq_diff[i, :, :, j:j + channel] = transform(diff_selected[:, :, stidx_diff + j, :])

        if flipable:
            flip = np.random.binomial(1, .5, 1)[0]
        else:
            flip = 0
        if flip == 1:
            seq = seq[:, ::-1]
            seq_same = seq_same[:, ::-1]
            seq_diff = seq_diff[:, ::-1]
    return seq, seq_same, seq_diff


def transform(image):
    return image/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


# def save_images(images, size, image_path):
#   return imsave(inverse_transform(images)*255., size, image_path)
#
# def imsave(images, size, path):
#   return scipy.misc.imsave(path, merge(images, size))

def FixedUnPooling(x, shape):
    """
    Unpool the input with a fixed mat to perform kronecker product with.
    :param input: NHWC tensor
    :param shape: int or [h, w]
    :returns: NHWC tensor
    """
    shape = shape2d(shape)

    # a faster implementation for this special case
    return UnPooling2x2ZeroFilled(x)

def UnPooling2x2ZeroFilled(x):
  out = tf.concat(axis=3, values=[x, tf.zeros_like(x)])
  out = tf.concat(axis=2, values=[out, tf.zeros_like(out)])

  sh = x.get_shape().as_list()
  if None not in sh[1:]:
    out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
    return tf.reshape(out, out_size)
  else:
    sh = tf.shape(x)
    return tf.reshape(out, [-1, sh[1] * 2, sh[2] * 2, sh[3]])

def shape2d(a):
  """
  a: a int or tuple/list of length 2
  """
  if type(a) == int:
      return [a, a]
  if isinstance(a, (list, tuple)):
      assert len(a) == 2
      return list(a)
  raise RuntimeError("Illegal shape: {}".format(a))

def resnet_backbone(x, num_blocks, train=True):
    l = resnet_group(x, 'EP_res_group0', resnet_basicblock, 64, num_blocks[0], 1, train=train)
    l = resnet_group(l, 'EP_res_group1', resnet_basicblock, 128, num_blocks[1], 2, train=train)
    l = resnet_group(l, 'EP_res_group2', resnet_basicblock, 256, num_blocks[2], 2, train=train)
    l = resnet_group(l, 'EP_res_group3', resnet_basicblock, 512, num_blocks[3], 2, train=train)
    return l


def resnet_group(l, name, block_func, features, count, stride, train=True):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('_block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1, train=train)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l

def resnet_basicblock(l, ch_out, stride, train=True):
    shortcut = tf.identity(l)
    l = tf.nn.relu(batch_norm(conv2d(l, ch_out, k_h=3, k_w=3, d_h=stride, d_w=stride, name="conv1", reuse=tf.AUTO_REUSE),
                         "conv1_bn1", train=train, reuse=tf.AUTO_REUSE))

    l = batch_norm(conv2d(l, ch_out, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2", reuse=tf.AUTO_REUSE),
                         "conv2_bn2", train=train, reuse=tf.AUTO_REUSE)
    return l + resnet_shortcut(shortcut, ch_out, stride, train=train)

def resnet_shortcut(l, n_out, stride, train=True):
    n_in = l.get_shape().as_list()[3]
    print n_in, n_out
    if n_in != n_out:   # change dimension when channel is not the same
        return batch_norm(conv2d(l, n_out, k_h=1, k_w=1, d_h=stride, d_w=stride, name="short_conv", reuse=tf.AUTO_REUSE),
                         "short_conv_bn2", train=train, reuse=tf.AUTO_REUSE)
    else:
        return l


def shape4d(a):
  # for use with tensorflow
  return [1] + shape2d(a) + [1]


def MaxPooling(x, shape, stride=None, padding='VALID'):
  """
  MaxPooling on images.
  :param input: NHWC tensor.
  :param shape: int or [h, w]
  :param stride: int or [h, w]. default to be shape.
  :param padding: 'valid' or 'same'. default to 'valid'
  :returns: NHWC tensor.
  """
  padding = padding.upper()
  shape = shape4d(shape)
  if stride is None:
    stride = shape
  else:
    stride = shape4d(stride)

  return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)