import os
import tensorflow as tf

from convlstm_cell import ConvGRUCell
from ops import *
from utils import *


class bi_convlstm_net(object):
    def __init__(self, image_size, batch_size=32, c_dim=3,
                 K=1, T=3, B=5, convlstm_layer_num=3, checkpoint_dir=None, dis_length = 16,
                 is_train=True, debug = False, reference_mode = "two", convlstm_kernel = [3, 3], dec = 'normal'):
        self.dec = dec
        self.convlstm_kernel = convlstm_kernel
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train
        self.dis_length=dis_length
        self.gf_dim = 32
        self.df_dim = 64

        self.c_dim = c_dim
        self.reference_mode = reference_mode
        self.K = K
        self.T = T
        self.B = B
        self.convlstm_layer_num = convlstm_layer_num
        self.forward_shape = [batch_size, B*(K+T)+K, self.image_size[0],
                           self.image_size[1], c_dim]

        self.backward_shape = [batch_size, B*(K+T)+K, self.image_size[0],
                           self.image_size[1], c_dim]

        self.xt_shape = [batch_size, self.image_size[0], self.image_size[1], c_dim]
        self.target_shape = [batch_size, self.image_size[0], self.image_size[1],
                             B*(K+T)+K, c_dim]
        self.debug = debug
        self.build_model()

    def build_model(self):
        self.forward_seq=tf.placeholder(tf.float32, self.forward_shape, name='forward_seq')
        self.target = tf.placeholder(tf.float32, self.target_shape, name='target')
        self.is_gen=tf.placeholder(tf.bool,name="is_gen")
        self.is_dis=tf.placeholder(tf.bool,name="is_dis")
        # pred: [batch * h * w * 1 * c_dim]
        pred = self.forward(self.forward_seq, self.target,self.is_gen)
        # self.G = tf.transpose(pred, [0, 2, 3, 1, 4])
        # self.G = batch * h * w * t * c_dim
        self.G = tf.concat(axis=3, values=pred)
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
            btarget = tf.reshape(self.target,
                                 [self.batch_size, self.image_size[0],
                                  self.image_size[1], -1])

            bgen = tf.reshape(self.G, [self.batch_size,
                                       self.image_size[0],
                                       self.image_size[1], -1])
            good_data = btarget
            gen_data = bgen
            full_length=btarget.get_shape().as_list()[3]
            ratio = 1
            if full_length > self.dis_length * self.c_dim:
                start = self.c_dim * random.randint(0, full_length / self.c_dim - self.dis_length)
                good_data = btarget[:,:,:,start:start+self.dis_length * self.c_dim]
                gen_data = bgen[:,:,:,start:start+self.dis_length * self.c_dim]
                ratio = full_length / (self.dis_length * self.c_dim *1.)
            if self.debug:
                print good_data.get_shape().as_list()
                print gen_data.get_shape().as_list()

            with tf.variable_scope("DIS", reuse=False):
                self.D, self.D_logits = self.discriminator(good_data,self.is_dis)

            with tf.variable_scope("DIS", reuse=True):
                self.D_, self.D_logits_ = self.discriminator(gen_data,self.is_dis)
            # pixel level loss
            self.L_p = tf.reduce_mean(
                tf.square(self.G - self.target)
            )
            # gradient loss
            self.L_gdl = gdl(gen_sim, true_sim, 1.)
            self.L_img = self.L_p + self.L_gdl

            self.d_loss_real = ratio * tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits, labels=tf.ones_like(self.D)
                )
            )
            self.d_loss_fake = ratio * tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.zeros_like(self.D_)
                )
            )
            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.L_GAN = ratio * tf.reduce_mean(
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

    def convRnn_seq_op(self, unit_index, seq, reuse=False):
        shape = seq.get_shape().as_list()[2:4]
        with tf.name_scope("for_convlstm"):
            with tf.variable_scope('for_convRnn_' + str(unit_index), reuse=reuse):
                for_cell = ConvGRUCell(shape, self.gf_dim, self.convlstm_kernel)
        with tf.name_scope("back_convlstm"):
            with tf.variable_scope('back_convRnn_' + str(unit_index), reuse=reuse):
                back_cell = ConvGRUCell(shape, self.gf_dim, self.convlstm_kernel)
        with tf.variable_scope('bidirection_rnn_' + str(unit_index), reuse=reuse):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(for_cell, back_cell, seq, dtype=seq.dtype)

        return tf.concat(outputs, 4)

    def forward(self, for_seq, seq_in, is_gen):
        reuse = False
        frames = []
        for_seq, res = self.pooling_feature_enc(for_seq, reuse=reuse)
        for i in range(self.convlstm_layer_num):
            for_seq = self.convRnn_seq_op(i, for_seq, reuse=reuse)
            # print for_seq.get_shape().as_list()
        for t in xrange(self.B * (self.T+self.K) + self.K):
            if self.dec == 'depool':
                x_hat = self.dec_cnn_depool(for_seq[:, t, :, :, :], reuse=reuse, train=is_gen)
            else:
                x_hat = self.dec_cnn(for_seq[:, t, :, :, :], reuse=reuse, train=is_gen)
            frames.append(tf.reshape(x_hat, [self.batch_size, self.image_size[0],
                                                 self.image_size[1], 1, self.c_dim]))
            reuse = True
        return frames

    def pooling_feature_enc(self, seq, reuse):
        feature=None
        res_in = []
        length = seq.get_shape().as_list()[1]
        for k in range(length):
            ref_frame = tf.reshape(seq[:, k, :, :, :],[-1, self.image_size[0], self.image_size[1],self.c_dim])

            conv1_1 = relu(conv2d(ref_frame,
                   output_dim=self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='fea_conv1_1', reuse=reuse))

            conv1_2 = relu(conv2d(conv1_1,
                   output_dim=self.gf_dim, k_h=3, k_w=3, d_h=2, d_w=2, name='fea_conv1_2', reuse=reuse))
            # conv1_2 128*128*32
            if k == (length-1):
                res_in.append(conv1_1)

            conv2_1 = relu(conv2d(conv1_2,
                   output_dim=self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1, name='fea_conv2_1', reuse=reuse))
            conv2_2 = relu(conv2d(conv2_1,
                   output_dim=self.gf_dim * 2, k_h=3, k_w=3, d_h=2, d_w=2, name='fea_conv2_2', reuse=reuse))
            # conv2_2 64*64*64
            if k == (length - 1):
                res_in.append(conv2_1)

            conv3_1 = relu(conv2d(conv2_2,
                   output_dim=self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1, name='fea_conv3_1', reuse=reuse))
            conv3_2 = relu(conv2d(conv3_1,
                   output_dim=self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1, name='fea_conv3_2', reuse=reuse))
            # conv3 32*32*128
            if k == (length - 1):
                res_in.append(conv3_1)

            if feature is None:
                feature=tf.reshape(conv3_2,[-1, 1, conv3_2.get_shape().as_list()[1],
                                            conv3_2.get_shape().as_list()[2], conv3_2.get_shape().as_list()[3]])
            else:
                feature=tf.concat([feature,
                                   tf.reshape(conv3_2,[-1, 1, conv3_2.get_shape().as_list()[1],
                                        conv3_2.get_shape().as_list()[2], conv3_2.get_shape().as_list()[3]])
                                   ], 1)
            reuse = True
        if self.debug:
            print "pooling_feature_enc,feature:{}".format(feature.get_shape())
        return feature, res_in

    # def encode_cnn(self, seq, reuse=False):
    def dec_cnn(self, comb, reuse=False, train=True):
        stride = int((self.image_size[0] / float(comb.get_shape().as_list()[1]))**(1/2.))
        if self.debug:
            print "dec_cnn comb:{}".format(comb.get_shape().as_list()[1])
            print "dec_cnn stride:{}".format(stride)
        shape1 = [self.batch_size, comb.get_shape().as_list()[1] * stride,
                  comb.get_shape().as_list()[2] * stride, self.gf_dim * 4]
        deconv1 = relu(batch_norm(deconv2d(comb,output_shape=shape1, k_h=4, k_w=4,
                      d_h=stride, d_w=stride, name='dec_deconv1', reuse=reuse), "dec_bn1",reuse=reuse,train=train))
        shape2 = [self.batch_size, deconv1.get_shape().as_list()[1] * stride,
                  deconv1.get_shape().as_list()[2] * stride, self.gf_dim * 2]
        deconv2 = relu(batch_norm(deconv2d(deconv1, output_shape=shape2, k_h=4, k_w=4,
                       d_h=stride, d_w=stride, name='dec_deconv2', reuse=reuse), "dec_bn2",reuse=reuse,train=train))
        shape3 = [self.batch_size, self.image_size[0],
                  self.image_size[1], self.gf_dim]
        deconv3 = relu(batch_norm(deconv2d(deconv2, output_shape=shape3, k_h=3, k_w=3,
                       d_h=1, d_w=1, name='dec_deconv3', reuse=reuse), "dec_bn3",reuse=reuse,train=train))
        shapeout = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]
        xtp1 = tanh(deconv2d(deconv3, output_shape=shapeout, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='dec_deconv_out', reuse=reuse))
        if self.debug:
            print "dec_cnn,xtp1:{}".format(xtp1.get_shape())
        return xtp1

    def dec_cnn_depool(self, comb, reuse=False, train=True):
        stride = int((self.image_size[0] / float(comb.get_shape().as_list()[1]))**(1/2.))
        if self.debug:
            print "dec_cnn comb:{}".format(comb.get_shape().as_list()[1])
            print "dec_cnn stride:{}".format(stride)
        shapel3 = [self.batch_size, comb.get_shape().as_list()[1] ,
                  comb.get_shape().as_list()[2] * stride, self.gf_dim * 4]
        shapeout3 = [self.batch_size, comb.get_shape().as_list()[1] ,
                  comb.get_shape().as_list()[2] * stride, self.gf_dim * 2]
        depool3 = FixedUnPooling(comb, [2, 2])
        deconv3_2 = relu(deconv2d(depool3, output_shape=shapel3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv3_2', reuse=reuse))
        deconv3_1 = relu(deconv2d(deconv3_2, output_shape=shapeout3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv3_1', reuse=reuse))
        shapeout3 = [self.batch_size, deconv3_1.get_shape().as_list()[1] * stride,
                     deconv3_1.get_shape().as_list()[2] * stride, self.gf_dim]
        depool2 = FixedUnPooling(deconv3_1, [2, 2])
        deconv2_1 = relu(deconv2d(depool2, output_shape=shapeout3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv2_1', reuse=reuse))
        shapeout1 = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]
        depool1 = FixedUnPooling(deconv2_1, [2, 2])
        xtp1 = tanh(deconv2d(depool1, output_shape=shapeout1, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='dec_deconv1_1', reuse=reuse))
        if self.debug:
            print "dec_cnn,xtp1:{}".format(xtp1.get_shape())
        return xtp1

    def discriminator(self, image, train=True):
        h0 = lrelu(conv2d(image, self.df_dim, name='dis_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, self.df_dim * 2, name='dis_h1_conv'),
                              "bn1",train=train))
        h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 4, name='dis_h2_conv'),
                              "bn2",train=train))
        h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 8, name='dis_h3_conv'),
                              "bn3",train=train))
        h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

        return tf.nn.sigmoid(h), h

    def save(self, sess, checkpoint_dir, step):
        model_name = "bi_conv_lstm.model"

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
