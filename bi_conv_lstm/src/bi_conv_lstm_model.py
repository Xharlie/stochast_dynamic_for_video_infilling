import os
import tensorflow as tf

from convlstm_cell import ConvLSTMCell
from ops import *
from utils import *


class bi_convlstm_net(object):
    def __init__(self, image_size, batch_size=32, c_dim=3,
                 K=1, T=3, B=5, convlstm_layer_num=3, checkpoint_dir=None,
                 is_train=True, debug = False, reference_mode = "two", convlstm_kernel = [3, 3]):

        self.convlstm_kernel = convlstm_kernel
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train

        self.gf_dim = 32
        self.df_dim = 64

        self.c_dim = c_dim
        self.reference_mode = reference_mode
        self.K = K
        self.T = T
        self.B = B
        self.convlstm_layer_num = convlstm_layer_num
        self.forward_shape = [batch_size, (B+1)*K, self.image_size[0],
                           self.image_size[1], c_dim]

        self.backward_shape = [batch_size, (B+1)*K, self.image_size[0],
                           self.image_size[1], c_dim]

        self.xt_shape = [batch_size, self.image_size[0], self.image_size[1], c_dim]
        self.target_shape = [batch_size, self.image_size[0], self.image_size[1],
                             B*(K+T)+K, c_dim]
        self.debug = debug
        self.build_model()

    def build_model(self):
        self.forward_seq=tf.placeholder(tf.float32, self.forward_shape, name='forward_seq')
        self.backward_seq=tf.placeholder(tf.float32, self.backward_shape, name='backward_seq')
        self.target = tf.placeholder(tf.float32, self.target_shape, name='target')
        pred = self.forward(self.forward_seq, self.backward_seq, self.target)
        self.G = tf.transpose(pred, [0, 2, 3, 1, 4])
        # self.G = tf.concat(axis=3, values=pred)
        if self.is_train:
            true_sim = inverse_transform(self.target[:, :, :, :, :])
            # change 1 channel by replicate to 3 channels
            if self.c_dim == 1: true_sim = tf.tile(true_sim, [1, 1, 1, 1, 3])
            # change axis to time, x, y, and put batch and time together
            true_sim = tf.reshape(tf.transpose(true_sim, [0, 3, 1, 2, 4]),
                                  [-1, self.image_size[0],
                                   self.image_size[1], 3])
            gen_sim = inverse_transform(self.G)
            # change 1 channel by replicate to 3 channels
            if self.c_dim == 1: gen_sim = tf.tile(gen_sim, [1, 1, 1, 1, 3])
            # change axis to time, x, y, and put batch and time together
            gen_sim = tf.reshape(tf.transpose(gen_sim, [0, 3, 1, 2, 4]),
                                 [-1, self.image_size[0],
                                  self.image_size[1], 3])
            # binput_f = tf.reshape(self.target[:, :, :, :self.K, :],
            #                     [self.batch_size, self.image_size[0],
            #                      self.image_size[1], -1])
            # binput_b = tf.reshape(self.target[:, :, :, self.K + self.T:, :],
            #                       [self.batch_size, self.image_size[0],
            #                        self.image_size[1], -1])
            btarget = tf.reshape(self.target[:, :, :, :, :],
                                 [self.batch_size, self.image_size[0],
                                  self.image_size[1], -1])
            bgen = tf.reshape(self.G, [self.batch_size,
                                       self.image_size[0],
                                       self.image_size[1], -1])
            good_data = btarget
            gen_data = bgen

            with tf.variable_scope("DIS", reuse=False):
                self.D, self.D_logits = self.discriminator(good_data)

            with tf.variable_scope("DIS", reuse=True):
                self.D_, self.D_logits_ = self.discriminator(gen_data)
            # pixel level loss
            self.L_p = tf.reduce_mean(
                tf.square(self.G - self.target[:, :, :, :, :])
            )
            # gradient loss
            self.L_gdl = gdl(gen_sim, true_sim, 1.)
            self.L_img = self.L_p + self.L_gdl

            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits, labels=tf.ones_like(self.D)
                )
            )
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.zeros_like(self.D_)
                )
            )
            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.L_GAN = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.ones_like(self.D_)
                )
            )

            self.loss_sum = tf.summary.scalar("L_img", self.L_img)
            self.L_p_sum = tf.summary.scalar("L_p", self.L_p)
            self.L_gdl_sum = tf.summary.scalar("L_gdl", self.L_gdl)
            self.L_GAN_sum = tf.summary.scalar("L_GAN", self.L_GAN)
            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
            self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

            self.t_vars = tf.trainable_variables()
            self.g_vars = [var for var in self.t_vars if 'DIS' not in var.name]
            self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
            num_param = 0.0
            for var in self.g_vars:
                num_param += int(np.prod(var.get_shape()));
            print("Number of parameters: %d" % num_param)
        self.saver = tf.train.Saver(max_to_keep=10)

    def convlstm_seq_op(self, unit_index, for_seq, back_seq, reuse=False):
        shape = for_seq.get_shape().as_list()[2:4]
        with tf.name_scope("for_convlstm"):
            with tf.variable_scope('for_convlstm_' + str(unit_index), reuse=reuse):
                for_cell = ConvLSTMCell(shape, self.gf_dim, self.convlstm_kernel)
                for_seq, state = tf.nn.dynamic_rnn(for_cell, for_seq, dtype=for_seq.dtype)
        with tf.name_scope("'back_convlstm"):
            with tf.variable_scope('back_convlstm_' + str(unit_index), reuse=reuse):
                back_cell = ConvLSTMCell(shape, self.gf_dim, self.convlstm_kernel)
                back_seq, state = tf.nn.dynamic_rnn(back_cell, back_seq, dtype=back_seq.dtype)
        with tf.name_scope("combine_convlstm"):
            for_seq = tf.concat([for_seq, tf.reverse(back_seq, [1])], 4)
            back_seq = tf.reverse(for_seq, [1])
        return for_seq, back_seq

    def forward(self, for_seq, back_seq, seq_in):
        reuse = False

        for i in range(self.convlstm_layer_num):
            for_seq, back_seq = self.convlstm_seq_op(i, for_seq, back_seq, reuse=reuse)

        return for_seq

    def discriminator(self, image):
        h0 = lrelu(conv2d(image, self.df_dim, name='dis_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name='dis_h1_conv'),
                              "bn1"))
        h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name='dis_h2_conv'),
                              "bn2"))
        h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 8, name='dis_h3_conv'),
                              "bn3"))
        h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

        return tf.nn.sigmoid(h), h

    def save(self, sess, checkpoint_dir, step):
        model_name = "MCNET.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir, model_name=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None: model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            print("     Loaded model: " + str(model_name))
            return True, model_name
        else:
            return False, None
