import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops
from convlstm_cell import *

from utils import *


def batch_norm(inputs, name, train=True, reuse=False):
  return tf.contrib.layers.batch_norm(inputs=inputs,is_training=train,
                                      reuse=reuse,scope=name,scale=True)


def conv2d(input_, output_dim, 
            k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
            name="conv2d", reuse=False, padding='SAME'):
   with tf.variable_scope(name, reuse=reuse):
     w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                         initializer=tf.contrib.layers.xavier_initializer())
     conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
 
     biases = tf.get_variable('biases', [output_dim],
                              initializer=tf.constant_initializer(0.0))
     conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
 
     return conv


def atrous_conv2d(input_, output_dim,
           rate=2, stddev=0.02,
           name="atrous_conv2d", reuse=False, padding='SAME'):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [rate, rate, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.atrous_conv2d(input_, w, rate=rate, padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def conv3d(input_, output_dim,
           k_d=1, k_h=3, k_w=3, s_d =1, s_h=1, s_w=1, stddev=0.02,
           name="conv3d", reuse=False, padding='SAME'):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv3d(input_, w, strides=[1, s_d, s_h, s_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

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
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv


def deconv3d(input_, output_shape,
             k_d, k_h=5, k_w=5, s_d = 2, s_h=1, s_w=1, stddev=0.02,
             name="deconv2d", reuse=False, with_w=False, padding='SAME'):
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1],
                                  input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        deconv = tf.nn.conv3d_transpose(input_, w,
                                        output_shape=output_shape,
                                        strides=[1, s_d, s_h, s_w, 1],
                                        padding=padding)

        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def relu(x):
  return tf.nn.relu(x)


def tanh(x):
  return tf.nn.tanh(x)


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


def shape4d(a):
  # for use with tensorflow
  return [1] + shape2d(a) + [1]


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


#@layer_register()
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


def gdl(gen_frames, gt_frames, alpha):
  """
  Calculates the sum of GDL losses between the predicted and gt frames.
  @param gen_frames: The predicted frames at each scale.
  @param gt_frames: The ground truth frames at each scale
  @param alpha: The power to which each gradient term is raised.
  @return: The GDL loss.
  """
  # create filters [-1, 1] and [[1],[-1]]
  # for diffing to the left and down respectively.
  pos = tf.constant(np.identity(3), dtype=tf.float32)
  neg = -1 * pos
  # [-1, 1]
  filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)
  # [[1],[-1]]
  filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])
  strides = [1, 1, 1, 1]  # stride of (1, 1)
  padding = 'SAME'

  gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
  gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
  gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
  gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))

  grad_diff_x = tf.abs(gt_dx - gen_dx)
  grad_diff_y = tf.abs(gt_dy - gen_dy)

  gdl_loss = tf.reduce_mean((grad_diff_x ** alpha + grad_diff_y ** alpha))

  # condense into one tensor and avg
  return gdl_loss


def linear(input_, output_size, name, stddev=0.02, bias_start=0.0, feat_axis = 1,
           reuse=False, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name, reuse=reuse):
        matrix = tf.get_variable("Matrix", [shape[feat_axis], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def linear_time(input_, output_size, name, stddev=0.02, bias_start=0.0, feat_axis = 2,
           reuse=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name, reuse=reuse):
        matrix = tf.get_variable("Matrix", [shape[feat_axis], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        h = tf.matmul(tf.reshape(input_, [-1, shape[feat_axis]]), matrix) + bias
        return tf.reshape(h,[-1, shape[1], output_size])

def conv_time(input_, output_size, name):
    shape = input_.get_shape().as_list()
    output = None
    for i in range(shape[1]):
        output_slice = lrelu(conv2d(input_[:,i,:,:,:], output_size,
                                         k_h=4, k_w=4, d_h=1, d_w=1, name=name + "1", reuse=tf.AUTO_REUSE))
        output_slice = tf.reshape(conv2d(output_slice, output_size,
                                         k_h=3, k_w=3, d_h=1, d_w=1, name=name + "2", reuse=tf.AUTO_REUSE),
                                  [shape[0], 1, shape[2], shape[3], output_size])
        if output is None:
            output = output_slice
        else:
            output = tf.concat([output, output_slice], axis = 1)
    return output

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


def kl_loss(batch_size, mu1, sigma1, mu2, sigma2):
    kld = tf.log(tf.div(sigma2, sigma1)) \
          + tf.div((tf.square(sigma1) + tf.square(tf.subtract(mu1, mu2))), (2 * tf.square(sigma2))) \
          - 1 / 2.0
    print("kld.shape", kld.get_shape().as_list())
    return tf.reduce_sum(kld) / batch_size

def serial_pose_encoder(self, seq, ref_seq, is_train=True, reuse=False):
    features = None
    res = None
    res_hidden = None
    for k in range(self.length):
        feature, res_list = pose_encoder(self, tf.reshape(seq[:, k, :, :, :],
            [-1, self.image_size[0], self.image_size[1], self.c_dim]),
            is_train=is_train, reuse=reuse)
        shape = feature.get_shape().as_list()
        feature = tf.reshape(feature, [-1, 1, shape[1], shape[2], shape[3]])
        if k == 0:
            features = feature
            if self.res_type == "first":
                res = res_list
                res_hidden = feature
        else:
            features = tf.concat([features, feature], 1)
        if k == self.length - 1 and self.res_type == "last":
            res = res_list
            res_hidden = feature
    if self.res_type == "avg":
        ref = tf.reduce_mean(ref_seq[:,self.B//2 : self.B//2+2,...], 1)
        res_hidden, res = pose_encoder(self, ref, is_train=is_train, reuse=reuse)
        res_hidden = tf.expand_dims(res_hidden, axis=1)
    return features, res, tf.squeeze(res_hidden, axis=1)

def serial_pose_encoder_noref(self, seq, is_train=True, reuse=False):
    features = None
    res = None
    res_hidden = None
    for k in range(self.length):
        feature, res_list = pose_encoder(self, tf.reshape(seq[:, k, :, :, :],
            [-1, self.image_size[0], self.image_size[1], self.c_dim]),
            is_train=is_train, reuse=reuse)
        shape = feature.get_shape().as_list()
        feature = tf.reshape(feature, [-1, 1, shape[1], shape[2], shape[3]])
        if k == 0:
            features = feature
            if self.res_type == "first":
                res = res_list
                res_hidden = feature
        else:
            features = tf.concat([features, feature], 1)
        if k == self.length - 1 and self.res_type == "last":
            res = res_list
            res_hidden = feature
    if self.res_type == "avg":
        ref = tf.reduce_mean(tf.concat([tf.expand_dims(seq[:,0,...], axis=1),
                                        tf.expand_dims(seq[:,-1,...], axis=1)], axis=1),
                             1)
        res_hidden, res = pose_encoder(self, ref, is_train=is_train, reuse=reuse)
        res_hidden = tf.expand_dims(res_hidden, axis=1)
    return features, res, tf.squeeze(res_hidden, axis=1)

def pose_encoder(self, x, is_train=True, reuse=tf.AUTO_REUSE):
    shape_type = x.get_shape().as_list()[1]
    h1 = dcgan_layer(x, self.ngf, "PE_1", is_train=is_train, reuse=reuse, bn=False)
    h2 = dcgan_layer(h1, self.ngf * 2, "PE_2", is_train=is_train, reuse=reuse)
    h3 = dcgan_layer(h2, self.ngf * 4, "PE_3", is_train=is_train, reuse=reuse)
    h4 = dcgan_layer(h3, self.ngf * 4, "PE_4", is_train=is_train, reuse=reuse)
    h5 = h4
    if shape_type == 128:
        print("encoder detect 128")
        h5 = dcgan_layer(h4, self.ngf * 4, "PE_5", is_train=is_train, reuse=reuse)
    print("h5 shape",h5.get_shape().as_list())
    padding = "VALID"
    if self.space_aware:
        padding = "SAME"
    h6 = tf.tanh(batch_norm(conv2d(h5, self.pattern_dim, k_h=4, k_w=4, d_h=1, d_w=1,
                name="PE_6", padding=padding, reuse=reuse), name="PE_6_bn", train=is_train, reuse=reuse))
    if self.normalize:
        h6 = tf.nn.l2_normalize(h6, 3)
    shape = h6.get_shape().as_list()
    print "h6 shape", shape
    return h6, [h1,h2,h3,h4,h5]

def post_gen_graph(self, post_zs):
    # 1:K+T
    predicted_frames = []
    post_predicted_frame_hiddens = []
    prediction_state = self.prediction_cell.zero_state(self.batch_size, tf.float32)
    for j in range(1, self.K + self.T):
        h_minus_one = self.inf_hidden_seq[:, j - 1, ...]
        post_predicted_frame_hidden, prediction_state = \
            self.frame_prediction(h_minus_one, post_zs[j - 1], prediction_state, is_train=True)
        predicted_frame = decoder(self, post_predicted_frame_hidden, self.res_list, self.res_hidden, is_train=True)
        predicted_frames.append(tf.expand_dims(predicted_frame, axis=1))
        post_predicted_frame_hiddens.append(tf.expand_dims(post_predicted_frame_hidden, axis=1))
    G_post = tf.reshape(tf.concat(predicted_frames, axis=1),
                   [self.batch_size, self.K + self.T - 1, self.image_size[0], self.image_size[1],
                    self.c_dim])
    post_predicted_frame_hiddens = tf.concat(post_predicted_frame_hiddens, axis=1)
    return G_post, post_predicted_frame_hiddens


def decoder(self, hidden, res, res_hidden, is_train = True, reuse = tf.AUTO_REUSE, all_layer_z = False):
    if self.res_ref == True:
        hidden = dcgan_layer(tf.concat([hidden, res_hidden], axis=3), self.ngf * 2,
                             "DE_combine", is_train=is_train, reuse=reuse, d_h=1, d_w=1)
    ratio = 4
    if self.space_aware:
        ratio = 1
    dec1 = dec_layer(self.batch_size, hidden, self.ngf * 4, "DE_d1_1", ratio= ratio, is_train = is_train, reuse=reuse)
    if res[0].get_shape().as_list()[1] == 64:
        dec1 = dec_layer(self.batch_size,tf.concat((dec1, res[4]), 3), self.ngf * 4, "DE_d1_2", is_train=is_train, reuse=reuse)
    dec2 = dec_layer(self.batch_size,tf.concat((dec1, res[3]), 3), self.ngf * 4, "DE_d2", is_train=is_train, reuse=reuse)
    dec3 = dec_layer(self.batch_size,tf.concat((dec2, res[2]), 3), self.ngf * 2, "DE_d3", is_train=is_train, reuse=reuse)
    dec4 = dec_layer(self.batch_size,tf.concat((dec3, res[1]), 3), self.ngf, "DE_d4", is_train=is_train, reuse=reuse)
    # dec5 = dec_layer(tf.concat((dec4, res[0]), 3), self.ngf, "EC_d5", is_train=is_train, reuse=reuse)
    dec5 = tf.nn.tanh(deconv2d(tf.concat((dec4, res[0]), 3),
               output_shape=[self.batch_size, self.image_size[0], self.image_size[1],self.c_dim],
               k_h=4, k_w=4, d_h=2, d_w=2, name="DE_d5", reuse=reuse))
    shape = dec5.get_shape().as_list()
    print "decoder h5 shape", shape
    return dec5

def post_gaussian_lstm(self, hidden_seq, is_gen=True, reuse=tf.AUTO_REUSE):
    # 0:K+T+1, leng = K+T+1
    length = hidden_seq.get_shape().as_list()[1]
    shape = hidden_seq.get_shape().as_list()[2:4]
    with tf.variable_scope('for_post_gaussian_lstm', reuse=reuse):
        # for_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.poseDim, state_is_tuple=True)
        if self.cell_type == "lstm":
            for_cell = ConvLSTMCell(shape, self.pattern_dim, self.convlstm_kernel)
        else:
            for_cell = ConvGRUCell(shape, self.pattern_dim, self.convlstm_kernel)
        seq, state = tf.nn.dynamic_rnn(for_cell, hidden_seq, dtype=hidden_seq.dtype)
    mus = []
    sigmas = []
    zs = []
    for i in range(1, length - 1):
        mu, sigma, z = get_spatial_distribution(seq[:, i, ...], self.z_dimension)
        mus.append(mu)
        sigmas.append(sigma)
        zs.append(z)
    return mus, sigmas, zs

def get_spatial_distribution(hidden, output_dim, reuse=tf.AUTO_REUSE):
    batch_shape = hidden.get_shape().as_list()[0]
    logvar = conv2d(hidden, output_dim,
        k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02,
        name="logvar_conv2d", padding='VALID', reuse=reuse)
    sigma = tf.exp(logvar*0.5)
    mu = conv2d(hidden, output_dim,
        k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02,
        name="mu_conv2d", padding='VALID', reuse=reuse)
    eps = tf.random_normal((batch_shape, 1, 1, output_dim), 0.0, 1.0, dtype=tf.float32)
    print ("sigma and eps shape:",sigma.get_shape().as_list(), eps.get_shape().as_list())
    z = tf.add(tf.multiply(sigma, eps), mu)
    return mu, sigma, z

def get_spatial_mean(hidden, output_dim, reuse=tf.AUTO_REUSE):
    mu = conv2d(hidden, output_dim,
                k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02,
                name="mu_conv2d", padding='VALID', reuse=reuse)
    return mu

def get_spatial_std(lstm_hidden, output_dim, reuse=tf.AUTO_REUSE):
    logvar = conv2d(lstm_hidden, output_dim,
                    k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02,
                    name="logvar_conv2d", padding='VALID', reuse=reuse)
    return tf.exp(logvar * 0.5)

def get_spatial_z(mu, sigma):
    # print "mu, sigma shape", sigma.get_shape().as_list(), mu.get_shape().as_list()
    batch_shape = sigma.get_shape().as_list()[0]
    output_dim = sigma.get_shape().as_list()[-1]
    eps = tf.random_normal((batch_shape, 1, 1, output_dim), 0.0, 1.0, dtype=tf.float32)
    print ("sigma and eps shape:", sigma.get_shape().as_list(), eps.get_shape().as_list())
    z = tf.add(tf.multiply(sigma, eps), mu)
    return z

def extract_pattern(self, ref_seq, is_train, reuse=tf.AUTO_REUSE):
    shape = ref_seq.get_shape().as_list()
    print("ref_seq shape:",shape)
    seq_transform = tf.reshape(tf.transpose(ref_seq, [0,2,3,4,1]),
                               [-1, shape[2], shape[3], shape[1] * shape[4]])
    h1 = dcgan_layer(seq_transform, self.ngf, "EXP_1", is_train=is_train, reuse=reuse)
    h2 = dcgan_layer(h1, self.ngf * 2, "EXP_2", is_train=is_train, reuse=reuse)
    h3 = dcgan_layer(h2, self.ngf * 4, "EXP_3", is_train=is_train, reuse=reuse)
    h4 = dcgan_layer(h3, self.ngf * 4, "EXP_4", is_train=is_train, reuse=reuse)
    h5 = h4
    if shape[2] == 128:
        print("encoder detect 128")
        h5 = dcgan_layer(h4, self.ngf * 4, "EXP_5", is_train=is_train, reuse=reuse)

    #  output dimension to be vector or 4*4 map
    if self.space_aware:
        padding = "SAME"
    else:
        padding = 'VALID'
    h6 = conv2d(h5, self.pattern_dim * 2, k_h=4, k_w=4, d_h=1, d_w=1, name="EXP_6", padding=padding, reuse=reuse)
    h6_shape = h6.get_shape().as_list()
    print("h6 shape", h6_shape)
    return h6[:,:,:,:self.pattern_dim], h6[:,:,:,self.pattern_dim:]

def dcgan_layer(input, nout, name, is_train=True, reuse=False, bn=True, d_h=2, d_w=2):
    if bn:
        return lrelu(batch_norm(conv2d(input, nout, d_h=d_h, d_w=d_w, name=name, reuse=reuse),
                                name + "_bn", train=is_train, reuse=reuse))
    else:
        return lrelu(conv2d(input, nout, d_h=d_h, d_w=d_w, name=name, reuse=reuse))

def dec_layer(batch_size, input, nout, name, is_train=True, ratio = 2, reuse=False):
    shape = input.get_shape().as_list()
    output_shape = [batch_size, shape[1] * ratio,
                    shape[2] * ratio, nout]
    return lrelu(batch_norm(deconv2d(input, output_shape = output_shape, k_h=4, k_w=4,
                  d_h=ratio, d_w=ratio, name=name, reuse=reuse), name + "_bn", train=is_train, reuse=reuse))


def create_mask(inf_seq, threshold = 0.1, weight = 3, negative_noise = False, c_dim = 1):
    shape = inf_seq.get_shape().as_list()
    weight_seq = []
    diff_binary_seq = []
    for i in range(0, shape[1] - 2):
        diff_map = tf.abs(tf.subtract(inf_seq[:,i,...], inf_seq[:,i+1,...]))
        if c_dim == 3:
            diff_binary = tf.expand_dims(tf.to_int32(tf.less(tf.constant(0),
                            tf.to_int32(tf.less(tf.constant(threshold), diff_map[:,:,:,0]))
                          + tf.to_int32(tf.less(tf.constant(threshold), diff_map[:,:,:,1]))
                          + tf.to_int32(tf.less(tf.constant(threshold), diff_map[:,:,:,2])))),axis=3)
        else:
            diff_binary = tf.to_int32(tf.less(tf.constant(threshold), diff_map))
        # print "diff_binary.shape", diff_binary.shape
        diff_convolve = tf.to_int32(tf.nn.convolution(tf.to_float(diff_binary),
                                                    tf.constant([[[[1.0]],[[1.0]],[[1.0]]],
                                                                 [[[1.0]],[[1.0]],[[1.0]]],
                                                                 [[[1.0]],[[1.0]],[[1.0]]]], dtype=tf.float32), "SAME"))
        diff_binary_positive = tf.multiply(diff_binary,
            tf.to_int32(tf.less_equal(tf.constant(2), diff_convolve)))
        diff_binary_negative = tf.multiply(diff_binary,
            tf.to_int32(tf.less(diff_convolve, tf.constant(2))))
        if(negative_noise):
            static_part = tf.subtract(1, diff_binary_negative)
        else:
            static_part = 1
        weight_map = tf.to_float(tf.add(tf.scalar_mul(weight, diff_binary_positive), static_part))
        sub_binary_map = tf.subtract(diff_binary_positive, diff_binary_negative)
        if c_dim == 3:
            weight_map = tf.tile(weight_map, [1,1,1,3])
            sub_binary_map = tf.tile(sub_binary_map, [1,1,1,3])
        weight_seq.append(tf.expand_dims(weight_map,axis=1))
        diff_binary_seq.append(tf.expand_dims(sub_binary_map,axis=1))
    return tf.concat(weight_seq, axis=1), tf.concat(diff_binary_seq, axis=1)

def save(model, sess, checkpoint_dir, step):
    model_name = "bi_conv_lstm.model"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

def load(model, sess, checkpoint_dir, model_name=""):
    print(" [*] Reading checkpoints... ")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        if model_name == "":
            model_name = ckpt_name
            print "ckpt_name", ckpt_name
        model.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
        print("     Loaded model: " + str(model_name))
        return True, model_name
    else:
        print ckpt
        return False, None
