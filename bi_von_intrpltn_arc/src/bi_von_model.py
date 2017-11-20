import os
import tensorflow as tf

from BasicConvLSTMCell import BasicConvLSTMCell
from ops import *
from utils import *


class bi_von_net(object):
    def __init__(self, image_size, batch_size=32, c_dim=3,
                 K=10, T=10, checkpoint_dir=None,
                 is_train=True, fea_enc_model='pooling',
                 dyn_enc_model='mix',
                 debug = False, reference_mode = "two"):

        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train

        self.gf_dim = 32
        self.df_dim = 64

        self.c_dim = c_dim
        self.reference_mode = reference_mode
        self.K = K
        self.T = T
        self.forward_shape = [batch_size, K, self.image_size[0],
                           self.image_size[1], c_dim]

        self.backward_shape = [batch_size, K, self.image_size[0],
                           self.image_size[1], c_dim]

        self.xt_shape = [batch_size, self.image_size[0], self.image_size[1], c_dim]
        self.target_shape = [batch_size, self.image_size[0], self.image_size[1],
                             2*K + T, c_dim]
        self.feature_enc = self.pooling_feature_enc
        if fea_enc_model == 'atrous':
            self.feature_enc = self.atrous_feature_enc
        self.dynamic_enc = self.mix_dynamic_enc
        if dyn_enc_model == 'deconv':
            self.dynamic_enc = self.deconv_dynamic_enc
        self.debug = debug
        self.build_model()

    def build_model(self):
        self.forward_seq=tf.placeholder(tf.float32, self.forward_shape, name='forward_seq')
        self.backward_seq=tf.placeholder(tf.float32, self.backward_shape, name='backward_seq')
        self.target = tf.placeholder(tf.float32, self.target_shape, name='target')
        pred = self.forward(self.forward_seq, self.backward_seq, self.target)
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

    def forward(self, for_seq, back_seq, seq_in):
        reuse = False

        # Forward and backward seq feature extraction

        # forward_feature is batchsize*K*128*128*64; for_res is residual
        forward_feature, for_res = self.feature_enc(for_seq, reuse=reuse)
        reuse = True

        backward_feature, back_res = self.feature_enc(back_seq, reuse=reuse)
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
        res = for_res
        if self.reference_mode == "two":
            for i in xrange(len(res)):
                res[i] = tf.concat(axis=3,values=[res[i], back_res[i]])
        for t in xrange(self.T):
            x_hat = self.dec_cnn(self.trans_cnn(dyn_bidrctn[:,t,:,:,:],
                                 forward_feature[:,-1,:,:,:], backward_feature[:,-1,:,:,:],reuse=reuse),
                                 res,reuse=reuse)
            gen_frames.append(tf.reshape(x_hat, [self.batch_size, self.image_size[0],
                                           self.image_size[1], 1, self.c_dim]))
            reuse = True
        return gen_frames

    def atrous_feature_enc(self, seq, reuse):
        feature=None
        res_in = []
        for k in range(self.K):
            ref_frame = tf.reshape(seq[:, k, :, :, :],[-1, self.image_size[0], self.image_size[1],self.c_dim])
            if self.debug:
                print "atrous_feature_enc, ref_frame:{}".format(feature.get_shape())
            conv1 = relu(atrous_conv2d(ref_frame,
                   output_dim=self.gf_dim, rate=1, name='fea_atrous_conv1', reuse=reuse))
            if k == (self.K-1):
                res_in.append(conv1)

            conv2 = relu(atrous_conv2d(conv1,
                   output_dim=self.gf_dim * 2, rate=2, name='fea_atrous_conv2', reuse=reuse))
            if k == (self.K-1):
                res_in.append(conv2)

            conv3 = relu(atrous_conv2d(conv2,
                   output_dim=self.gf_dim * 2, rate=4, name='fea_atrous_conv3', reuse=reuse))
            if k == (self.K-1):
                res_in.append(conv3)

            conv4 = relu(atrous_conv2d(conv3,
                   output_dim=self.gf_dim * 4, rate=8, name='fea_atrous_conv4', reuse=reuse))
            if feature is None:
                feature=tf.reshape(conv4,[-1, 1, self.image_size[0], self.image_size[1],self.gf_dim * 2])
                if self.debug:
                    print "atrous_feature_enc, feature reshape:{}".format(feature.get_shape())
            else:
                feature=tf.concat([feature,
                                   tf.reshape(conv4,[-1, 1, self.image_size[0], self.image_size[1],self.gf_dim * 2])
                                   ], 1)
            reuse = True
            if self.debug:
                print "atrous_feature_enc, feature:{}".format(feature.get_shape())
        return feature, res_in


    def pooling_feature_enc(self, seq, reuse):
        feature=None
        res_in = []
        for k in range(self.K):

            ref_frame = tf.reshape(seq[:, k, :, :, :],[-1, self.image_size[0], self.image_size[1],self.c_dim])

            conv1 = relu(conv2d(ref_frame,
                   output_dim=self.gf_dim, k_h=5, k_w=5, d_h=1, d_w=1, name='fea_conv1', reuse=reuse))
            # conv1 128*128*32
            if k == (self.K-1):
                res_in.append(conv1)

            pool1 = MaxPooling(conv1, [2, 2])

            conv2 = relu(conv2d(pool1,
                   output_dim=self.gf_dim * 2, k_h=5, k_w=5, d_h=1, d_w=1, name='fea_conv2', reuse=reuse))
            # conv2 64*64*64
            if k == (self.K - 1):
                res_in.append(conv2)

            pool2 = MaxPooling(conv2, [2, 2])

            conv3 = relu(conv2d(pool2,
                   output_dim=self.gf_dim * 4, k_h=7, k_w=7, d_h=1, d_w=1, name='fea_conv3', reuse=reuse))
            # conv3 32*32*128
            if k == (self.K - 1):
                res_in.append(conv3)

            pool3 = MaxPooling(conv3, [2, 2])

            if feature is None:
                feature=tf.reshape(pool3,[-1, 1, pool3.get_shape().as_list()[1],
                                          pool3.get_shape().as_list()[2], pool3.get_shape().as_list()[3]])
            else:
                feature=tf.concat([feature,
                                   tf.reshape(pool3,[-1, 1, pool3.get_shape().as_list()[1],
                                          pool3.get_shape().as_list()[2], pool3.get_shape().as_list()[3]])
                                   ], 1)
            reuse = True
        if self.debug:
            print "pooling_feature_enc,feature:{}".format(feature.get_shape())
        return feature, res_in

    def deconv_dynamic_enc(self, fea, reuse=False):
        shape1 = [fea.get_shape().as_list()[0], self.K, fea.get_shape().as_list()[2],
                  fea.get_shape().as_list()[3], self.gf_dim*2]
        deconv1 = relu(deconv3d(fea, output_shape = shape1, k_d = 3, k_h=3, k_w=3,
                              s_d = 1, s_h=1, s_w=1, name='dyn_deconv1', reuse=reuse))
        shape2 = [fea.get_shape().as_list()[0], self.K, deconv1.get_shape().as_list()[2],
                  deconv1.get_shape().as_list()[3], self.gf_dim*2]
        deconv2 = relu(deconv3d(deconv1, output_shape=shape2, k_d=3, k_h=3, k_w=3,
                              s_d = 1, s_h=1, s_w=1, name='dyn_deconv2', reuse=reuse))
        shape3 = [fea.get_shape().as_list()[0], self.T, deconv2.get_shape().as_list()[3],
                  deconv2.get_shape().as_list()[3], self.gf_dim]
        deconv3 = relu(deconv3d(deconv2, output_shape=shape3, k_d=3, k_h=3, k_w=3,
                              s_d = self.T / self.K, s_h=1, s_w=1, name='dyn_deconv3', reuse=reuse))
        shape4 = [fea.get_shape().as_list()[0], self.T, deconv3.get_shape().as_list()[3],
                  deconv3.get_shape().as_list()[3], self.gf_dim]
        deconv4 = relu(deconv3d(deconv3, output_shape=shape4, k_d=3, k_h=3, k_w=3,
                              s_d = 1, s_h=1, s_w=1, name='dyn_deconv4', reuse=reuse))
        return deconv4

    def mix_dynamic_enc(self, fea, reuse=False):
        conv1 = relu(conv3d(fea, self.gf_dim, k_d = self.K, k_h=3, k_w=3,
                              s_d = 1, s_h=1, s_w=1, name='dyn_conv1', reuse=reuse, padding='SAME'))
        conv2 = relu(conv3d(conv1, self.gf_dim, k_d=self.K, k_h=3, k_w=3,
                              s_d = 1, s_h=1, s_w=1, name='dyn_conv2', reuse=reuse, padding='SAME'))
        shape3 = [fea.get_shape().as_list()[0], self.T, conv2.get_shape().as_list()[2],
                  conv2.get_shape().as_list()[3], self.gf_dim]
        deconv3 = relu(deconv3d(conv2, output_shape=shape3, k_d=3, k_h=3, k_w=3,
                              s_d = self.T / self.K, s_h=1, s_w=1, name='dyn_deconv3', reuse=reuse))
        shape4 = [fea.get_shape().as_list()[0], self.T, deconv3.get_shape().as_list()[2],
                  deconv3.get_shape().as_list()[3], self.gf_dim]
        deconv4 = relu(deconv3d(deconv3, output_shape=shape4, k_d=3, k_h=3, k_w=3,
                              s_d = 1, s_h=1, s_w=1, name='dyn_deconv4', reuse=reuse))
        if self.debug:
            print "mix_dynamic_enc,deconv4:{}".format(deconv4.get_shape())
        return deconv4

    def trans_cnn(self, frame_tran, fxt_fea, bxt_fea, reuse=False):
        # frame_tran = tf.reshape(frame_tran,
        #                         [frame_tran.get_shape().as_list()[0],
        #                          frame_tran.get_shape().as_list()[1],
        #                          frame_tran.get_shape().as_list()[2],
        #                          frame_tran.get_shape().as_list()[-1]])
        if self.reference_mode == "two":
            combine = tf.concat([fxt_fea, frame_tran, bxt_fea], 3)
        else:
            combine = tf.concat([fxt_fea, frame_tran], 3)
        trans1 = relu(conv2d(combine,
                            output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                            d_h=1, d_w=1, name='trans_conv1', reuse=reuse))
        trans2 = relu(conv2d(trans1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                            d_h=1, d_w=1, name='trans_conv2', reuse=reuse))
        trans3 = relu(conv2d(trans2, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='trans_conv3', reuse=reuse))
        if self.debug:
            print "trans_cnn,trans3:{}".format(trans3.get_shape())
        return trans3

    def dec_cnn(self, comb, res, reuse=False):
        stride = int((self.image_size[0] / float(comb.get_shape().as_list()[1]))**(1/3.))
        if self.debug:
            print "dec_cnn comb:{}".format(comb.get_shape().as_list()[1])
            print "dec_cnn stride:{}".format(stride)
        shape1 = [self.batch_size, comb.get_shape().as_list()[1] * stride,
                  comb.get_shape().as_list()[2] * stride, self.gf_dim * 4]
        deconv1 = relu(batch_norm(tf.concat([deconv2d(comb,output_shape=shape1, k_h=3, k_w=3,
                      d_h=stride, d_w=stride, name='dec_deconv1', reuse=reuse), res[2]],axis=3), "dec_bn1",reuse=reuse))
        shape2 = [self.batch_size, deconv1.get_shape().as_list()[1] * stride,
                  deconv1.get_shape().as_list()[2] * stride, self.gf_dim * 2]
        deconv2 = relu(batch_norm(tf.concat([deconv2d(deconv1, output_shape=shape2, k_h=3, k_w=3,
                       d_h=stride, d_w=stride, name='dec_deconv2', reuse=reuse), res[1]],axis=3), "dec_bn2",reuse=reuse))
        shape3 = [self.batch_size, self.image_size[0],
                  self.image_size[1], self.gf_dim]
        deconv3 = relu(batch_norm(tf.concat([deconv2d(deconv2, output_shape=shape3, k_h=3, k_w=3,
                       d_h=stride, d_w=stride, name='dec_deconv3', reuse=reuse), res[0]],axis=3), "dec_bn3",reuse=reuse))
        shapeout = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]
        xtp1 = tanh(deconv2d(deconv3, output_shape=shapeout, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='dec_deconv_out', reuse=reuse))
        if self.debug:
            print "dec_cnn,xtp1:{}".format(xtp1.get_shape())
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
