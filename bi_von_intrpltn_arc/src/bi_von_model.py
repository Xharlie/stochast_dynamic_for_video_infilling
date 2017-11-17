import os
import tensorflow as tf

from BasicConvLSTMCell import BasicConvLSTMCell
from ops import *
from utils import *


class bi_von_net(object):
    def __init__(self, image_size, batch_size=32, c_dim=3,
                 K=10, T=10, checkpoint_dir=None, is_train=True):

        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train

        self.gf_dim = 32
        self.df_dim = 64

        self.c_dim = c_dim
        self.K = K
        self.T = T
        self.forward_shape = [batch_size, K, self.image_size[0],
                           self.image_size[1], c_dim]

        self.backward_shape = [batch_size, K, self.image_size[0],
                           self.image_size[1], c_dim]

        self.xt_shape = [batch_size, self.image_size[0], self.image_size[1], c_dim]
        self.target_shape = [batch_size, self.image_size[0], self.image_size[1],
                             2*K + T, c_dim]

        self.build_model()

    def build_model(self):
        self.forward_seq=tf.placeholder(tf.float32, self.forward_shape, name='forward_seq')
        self.backward_seq=tf.placeholder(tf.float32, self.backward_shape, name='backward_seq')
        self.target = tf.placeholder(tf.float32, self.target_shape, name='target')
        self.fxt = tf.placeholder(tf.float32, self.xt_shape, name='fxt')
        self.bxt = tf.placeholder(tf.float32, self.xt_shape, name='bxt')
        pred = self.forward(self.forward_seq, self.backward_seq, self.target, self.fxt, self.bxt)
        self.G = tf.concat(axis=3, values=pred)
        if self.is_train:
            true_sim = inverse_transform(self.target[:, :, :, self.K:self.K + self.T, :])
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
            binput_f = tf.reshape(self.target[:, :, :, :self.K, :],
                                [self.batch_size, self.image_size[0],
                                 self.image_size[1], -1])
            binput_b = tf.reshape(self.target[:, :, :, self.K + self.T:, :],
                                  [self.batch_size, self.image_size[0],
                                   self.image_size[1], -1])
            btarget = tf.reshape(self.target[:, :, :, self.K:self.K + self.T, :],
                                 [self.batch_size, self.image_size[0],
                                  self.image_size[1], -1])
            bgen = tf.reshape(self.G, [self.batch_size,
                                       self.image_size[0],
                                       self.image_size[1], -1])

            good_data = tf.concat(axis=3, values=[binput_f, btarget, binput_b])
            gen_data = tf.concat(axis=3, values=[binput_f, bgen, binput_b])

            with tf.variable_scope("DIS", reuse=False):
                self.D, self.D_logits = self.discriminator(good_data)

            with tf.variable_scope("DIS", reuse=True):
                self.D_, self.D_logits_ = self.discriminator(gen_data)
            # pixel level loss
            self.L_p = tf.reduce_mean(
                tf.square(self.G - self.target[:, :, :, self.K:self.K+self.T, :])
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

    def forward(self, for_seq, back_seq, seq_in, fxt, bxt):
        reuse = False

        # Forward and backward seq feature extraction

        # forward_feature is batchsize*K*128*128*256
        forward_feature = self.feature_enc(for_seq, reuse=reuse)
        reuse = True

        backward_feature = self.feature_enc(back_seq, reuse=reuse)
        # print backward_feature.get_shape()
        # forward and backward seq dynamic 3d convolution:
        # forward_dyn is batchsize*T*128*128*32
        reuse = False
        forward_dyn = self.dynamic_enc(forward_feature, reuse=reuse)
        reuse = True
        backward_dyn = self.dynamic_enc(backward_feature, reuse=reuse)
        # print backward_dyn.get_shape()

        # combine forward_dyn and reverse of backward_dyn, to get bidirectional dynamic
        # dyn_bidrctn is batchsize*T*128*128*64
        dyn_bidrctn = tf.concat([forward_dyn, tf.reverse(backward_dyn,[1])], 4)

        # generate frames
        # gen_frames is batchsize*128*128*T*3
        gen_frames = []
        reuse = False
        for t in xrange(self.T):
            x_hat = self.trans_cnn(dyn_bidrctn[:,t,:,:,:], fxt, bxt,reuse=reuse)
            gen_frames.append(tf.reshape(x_hat, [self.batch_size, self.image_size[0],
                                           self.image_size[1], 1, self.c_dim]))
            reuse = True
        return gen_frames

    def feature_enc(self, seq, reuse):
        feature=None
        for k in range(self.K):
            ref_frame = tf.reshape(seq[:, k, :, :, :],[-1, self.image_size[0], self.image_size[1],self.c_dim])
            conv1 = relu(atrous_conv2d(ref_frame,
                   output_dim=self.gf_dim, rate=1, name='fea_atrous_conv1', reuse=reuse))

            conv2 = relu(atrous_conv2d(conv1,
                   output_dim=self.gf_dim * 2, rate=2, name='fea_atrous_conv2', reuse=reuse))

            conv3 = relu(atrous_conv2d(conv2,
                   output_dim=self.gf_dim * 2, rate=4, name='fea_atrous_conv3', reuse=reuse))

            conv4 = relu(atrous_conv2d(conv3,
                   output_dim=self.gf_dim * 4, rate=8, name='fea_atrous_conv4', reuse=reuse))
            if feature is None:
                feature=tf.reshape(conv4,[-1, 1, self.image_size[0], self.image_size[1],self.gf_dim * 4])
            else:
                feature=tf.concat([feature,
                                   tf.reshape(conv4,[-1, 1, self.image_size[0], self.image_size[1],self.gf_dim * 4])
                                   ], 1)
            reuse = True
        return feature

    def dynamic_enc(self, seq, reuse=False):
        # TODO change first dimension dynamically
        shape1 = [self.batch_size, self.K, self.image_size[0], self.image_size[1], self.gf_dim]
        deconv1 = relu(deconv3d(seq, output_shape = shape1, k_d = 2, k_h=3, k_w=3,
                              s_d = 1, s_h=1, s_w=1, name='dyn_deconv1', reuse=reuse))
        shape2 = [self.batch_size, self.K, self.image_size[0], self.image_size[1], self.gf_dim]
        deconv2 = relu(deconv3d(deconv1, output_shape=shape2, k_d=3, k_h=3, k_w=3,
                              s_d = 1, s_h=1, s_w=1, name='dyn_deconv2', reuse=reuse))
        shape3 = [self.batch_size, self.T, self.image_size[0], self.image_size[1], self.gf_dim]
        deconv3 = relu(deconv3d(deconv2, output_shape=shape3, k_d=3, k_h=3, k_w=3,
                              s_d = self.T / self.K, s_h=1, s_w=1, name='dyn_deconv3', reuse=reuse))
        shape4 = [self.batch_size, self.T, self.image_size[0], self.image_size[1], self.gf_dim]
        deconv4 = relu(deconv3d(deconv3, output_shape=shape4, k_d=2, k_h=3, k_w=3,
                              s_d = 1, s_h=1, s_w=1, name='dyn_deconv4', reuse=reuse))
        return deconv4

    # def content_enc(self, xt, reuse):
    #     res_in = []
    #     conv1_1 = relu(conv2d(xt, output_dim=self.gf_dim, k_h=3, k_w=3,
    #                           d_h=1, d_w=1, name='cont_conv1_1', reuse=reuse))
    #     conv1_2 = relu(conv2d(conv1_1, output_dim=self.gf_dim, k_h=3, k_w=3,
    #                           d_h=1, d_w=1, name='cont_conv1_2', reuse=reuse))
    #     res_in.append(conv1_2)
    #     pool1 = MaxPooling(conv1_2, [2, 2])
    #
    #     conv2_1 = relu(conv2d(pool1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
    #                           d_h=1, d_w=1, name='cont_conv2_1', reuse=reuse))
    #     conv2_2 = relu(conv2d(conv2_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
    #                           d_h=1, d_w=1, name='cont_conv2_2', reuse=reuse))
    #     res_in.append(conv2_2)
    #     pool2 = MaxPooling(conv2_2, [2, 2])
    #
    #     conv3_1 = relu(conv2d(pool2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
    #                           d_h=1, d_w=1, name='cont_conv3_1', reuse=reuse))
    #     conv3_2 = relu(conv2d(conv3_1, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
    #                           d_h=1, d_w=1, name='cont_conv3_2', reuse=reuse))
    #     conv3_3 = relu(conv2d(conv3_2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
    #                           d_h=1, d_w=1, name='cont_conv3_3', reuse=reuse))
    #     res_in.append(conv3_3)
    #     pool3 = MaxPooling(conv3_3, [2, 2])
    #     return pool3, res_in
    #
    # def comb_layers(self, h_dyn, h_cont, reuse=False):
    #     comb1 = relu(conv2d(tf.concat(axis=3, values=[h_dyn, h_cont]),
    #                         output_dim=self.gf_dim * 4, k_h=3, k_w=3,
    #                         d_h=1, d_w=1, name='comb1', reuse=reuse))
    #     comb2 = relu(conv2d(comb1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
    #                         d_h=1, d_w=1, name='comb2', reuse=reuse))
    #     h_comb = relu(conv2d(comb2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
    #                          d_h=1, d_w=1, name='h_comb', reuse=reuse))
    #     return h_comb
    #
    # def residual(self, input_dyn, input_cont, reuse=False):
    #     n_layers = len(input_dyn)
    #     res_out = []
    #     for l in xrange(n_layers):
    #         input_ = tf.concat(axis=3, values=[input_dyn[l], input_cont[l]])
    #         out_dim = input_cont[l].get_shape()[3]
    #         res1 = relu(conv2d(input_, output_dim=out_dim,
    #                            k_h=3, k_w=3, d_h=1, d_w=1,
    #                            name='res' + str(l) + '_1', reuse=reuse))
    #         res2 = conv2d(res1, output_dim=out_dim, k_h=3, k_w=3,
    #                       d_h=1, d_w=1, name='res' + str(l) + '_2', reuse=reuse)
    #         res_out.append(res2)
    #     return res_out

    def trans_cnn(self, frame_tran, fxt, bxt, reuse=False):

        frame_tran=tf.reshape(frame_tran,
           [frame_tran.get_shape().as_list()[0],self.image_size[0],self.image_size[1],
            frame_tran.get_shape().as_list()[-1]])
        dc_combine = tf.concat([fxt, frame_tran, bxt], 3)
        shape1 = [self.batch_size, self.image_size[0],
                   self.image_size[1], self.gf_dim * 3]
        deconv1 = relu(batch_norm(deconv2d(dc_combine,output_shape=shape1, k_h=3, k_w=3,
                      d_h=1, d_w=1, name='dec_deconv1', reuse=reuse), "trans_bn1",reuse=reuse))
        shape2 = [self.batch_size, self.image_size[0],
                  self.image_size[1], self.gf_dim * 3]
        deconv2 = relu(batch_norm(deconv2d(deconv1, output_shape=shape1, k_h=3, k_w=3,
                       d_h=1, d_w=1, name='dec_deconv2', reuse=reuse),"trans_bn2",reuse=reuse))
        shape3 = [self.batch_size, self.image_size[0],
                  self.image_size[1], self.gf_dim * 3]
        deconv3 = relu(batch_norm(deconv2d(deconv2, output_shape=shape1, k_h=3, k_w=3,
                       d_h=1, d_w=1, name='dec_deconv3', reuse=reuse),"trans_bn3",reuse=reuse))
        shapeout = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]
        xtp1 = tanh(deconv2d(deconv3, output_shape=shapeout, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='dec_deconv_out', reuse=reuse))
        return xtp1

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
