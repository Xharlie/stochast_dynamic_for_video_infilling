import os
import tensorflow as tf

from convlstm_cell import ConvGRUCell
from ops import *
from utils import *


class bi_convlstm_net(object):
    def __init__(self, image_size, batch_size=32, c_dim=3,
                 K=1, T=3, B=5, convlstm_layer_num=3, checkpoint_dir=None, dis_length = 11,
                 is_train=True, debug = False, reference_mode = "two", pixel_loss="l2",
                 convlstm_kernel = [3, 3], dec = 'normal', model="bi_pass", Unet = True, use_gt=True,res_mode="mix"):
        self.dec = dec
        self.res_mode = res_mode
        self.use_gt = use_gt
        self.pixel_loss = pixel_loss
        self.convlstm_kernel = convlstm_kernel
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train
        self.dis_length=dis_length
        self.gf_dim = 32
        self.df_dim = 64
        self.counter = 0
        self.c_dim = c_dim
        self.reference_mode = reference_mode
        self.K = K
        self.T = T
        self.B = B
        self.length = B*(K+T)+K
        self.convlstm_layer_num = convlstm_layer_num
        self.forward_shape = [batch_size, B*(K+T)+K, self.image_size[0],
                           self.image_size[1], c_dim]

        self.backward_shape = [batch_size, B*(K+T)+K, self.image_size[0],
                           self.image_size[1], c_dim]

        self.xt_shape = [batch_size, self.image_size[0], self.image_size[1], c_dim]
        self.target_shape = [batch_size, self.image_size[0], self.image_size[1],
                             B*(K+T)+K, c_dim]
        self.debug = debug
        self.model = model
        self.Unet = Unet
        self.build_model()

    def build_model(self):
        self.forward_seq=tf.placeholder(tf.float32, self.forward_shape, name='forward_seq')
        self.target = tf.placeholder(tf.float32, self.target_shape, name='target')
        self.is_gen=tf.placeholder(tf.bool,name="is_gen")
        self.is_dis=tf.placeholder(tf.bool,name="is_dis")
        self.conv_layer_weight=tf.placeholder(tf.float32, [self.convlstm_layer_num], name="conv_layer_weight")

        self.conv_layer_index= tf.placeholder(tf.int32, name="conv_layer_index")
        self.loss_reduce_weight= tf.placeholder(tf.float32, name="loss_reduce_weight")
        # pred: [batch * h * w * 1 * c_dim]
        if self.model == "bi_pass":
            pred = self.forward(self.forward_seq, self.is_gen)
            self.G = tf.concat(axis=3, values=pred)
            self.normal_G = self.G
            self.target_deligate = self.target
            self.normal_target_deligate = self.target
        else:
            self.G, self.normal_G = self.forward_progressively(self.forward_seq, self.is_gen)
            self.target_deligate, self.normal_target_deligate = self.get_target_frames(self.target)

        if self.is_train:
            true_sim = inverse_transform(self.target_deligate[:, :, :, :, :])
            # change 1 channel by replicate to 3 channels
            if self.c_dim == 1: true_sim = tf.tile(true_sim, [1, 1, 1, 1, 3])
            # change axis to time, x, y, and put batch and time together
            true_sim = tf.reshape(tf.transpose(true_sim, [0, 3, 1, 2, 4]),
                                  [-1, self.image_size[0], self.image_size[1], 3])
            # change -1 - 1  to  0 - 1
            gen_sim = inverse_transform(self.G)
            # change 1 channel by replicate to 3 channels
            if self.c_dim == 1: gen_sim = tf.tile(gen_sim, [1, 1, 1, 1, 3])
            # change axis to time, x, y, and put batch and time together
            gen_sim = tf.reshape(tf.transpose(gen_sim, [0, 3, 1, 2, 4]),
                                 [-1, self.image_size[0],
                                  self.image_size[1], 3])
            btarget = tf.reshape(self.normal_target_deligate,
                                 [self.batch_size, self.image_size[0],
                                  self.image_size[1], -1])
            bgen = tf.reshape(self.normal_G, [self.batch_size,
                                       self.image_size[0],
                                       self.image_size[1], -1])
            full_length = 11
            # if full_length >= self.dis_length * self.c_dim:
            start = self.c_dim * random.randint(0, full_length - self.dis_length)
            print start
            good_data = tf.reshape(btarget[:,:,:,start:start+self.dis_length * self.c_dim],[self.batch_size, self.image_size[0],
                                  self.image_size[1], self.dis_length * self.c_dim])
            gen_data = tf.reshape(bgen[:,:,:,start:start+self.dis_length * self.c_dim],[self.batch_size, self.image_size[0],
                                  self.image_size[1], self.dis_length * self.c_dim])
            ratio = 1 #full_length / (self.dis_length * self.c_dim *1.)
            if self.debug:
                print good_data.get_shape().as_list()
                print gen_data.get_shape().as_list()

            with tf.variable_scope("DIS", reuse=False):
                self.D, self.D_logits = self.discriminator(good_data,self.is_dis)

            with tf.variable_scope("DIS", reuse=True):
                self.D_, self.D_logits_ = self.discriminator(gen_data,self.is_dis)
            # pixel level loss
            if self.pixel_loss == "l2":
                self.L_p = tf.reduce_mean(
                    tf.square(self.G - self.target_deligate)
                )
            else:
                self.L_p = tf.reduce_mean(
                    tf.abs(self.G - self.target_deligate)
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

            self.L_img_sum = tf.summary.scalar("L_img", self.L_img)
            self.L_p_sum = tf.summary.scalar("L_p", self.L_p)
            self.L_gdl_sum = tf.summary.scalar("L_gdl", self.L_gdl)
            self.L_GAN_sum = tf.summary.scalar("L_GAN", self.L_GAN)
            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

            self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

            self.t_vars = tf.trainable_variables()
            self.g_vars = [var for var in self.t_vars if 'DIS' not in var.name]
            self.g_2_vars = [var for var in self.t_vars if '_convRnn_1' in var.name]
            self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
            num_param = 0.0

            recon_size = 3
            self.train_seq_img = tf.placeholder(tf.float32, [None,
                             self.image_size[0] * 3 * recon_size,
                             self.image_size[1] * (self.B*(self.K+self.T)+self.K), self.c_dim])
            self.test_seq_img = tf.placeholder(tf.float32, [None,
                             self.image_size[0] * 3 * recon_size,
                             self.image_size[1] * (self.B*(self.K+self.T)+self.K), self.c_dim])

            self.train_seq_img_summary = tf.summary.image('train_seq_img', self.train_seq_img)
            self.test_seq_img_summary = tf.summary.image('test_seq_img', self.test_seq_img)
            self.summary_merge_seq_img = tf.summary.merge([
                self.train_seq_img_summary, self.test_seq_img_summary])
            for var in self.g_vars:
                num_param += int(np.prod(var.get_shape()))
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

    def forward(self, for_seq, is_gen):
        reuse = False
        frames = []
        for_seq, res = self.pooling_feature_enc(for_seq, reuse=reuse)
        for_seq = self.convRnn_seq_op(0, for_seq, reuse=reuse)
            # print for_seq.get_shape().as_list()
        for t in xrange(self.B * (self.T+self.K) + self.K):
            if self.dec == 'depool':
                res_t = 0 if self.res_mode == "mix" or self.res_mode == "one" else t
                x_hat = self.dec_cnn_depool(for_seq[:, t, :, :, :], res[res_t], reuse=reuse, train=is_gen)
            else:
                x_hat = self.dec_cnn(for_seq[:, t, :, :, :], reuse=reuse, train=is_gen)
            frames.append(tf.reshape(x_hat, [self.batch_size, self.image_size[0],
                                                 self.image_size[1], 1, self.c_dim]))
            reuse = True
        return frames

    def conditional_use_gt(self, seq, index, original_seq):
        if self.use_gt:
            grounded_seq = tf.expand_dims(original_seq[:, 0, :, :, :], 1)
            section = 2 ** (index+1)
            for t in xrange(1, self.B + self.K):
                seq_sec = seq[:, (t-1) * section + 1: t  * section, :, :, :]
                # if (section==2): seq_sec = tf.expand_dims(seq_sec,1)
                grounded_seq = tf.concat([grounded_seq, seq_sec, tf.expand_dims(original_seq[:,t,:,:,:],1)], axis=1)
            return grounded_seq
        else:
            return seq

    def convRnn_progressive_op(self, unit_index, original_seq, seq, reuse=tf.AUTO_REUSE):
        # print unit_index
        shape = seq.get_shape().as_list()[2:4]
        if self.model == 'bi_serial_progressive':
            scopename = 'for_convRnn_' + str(unit_index)
            with tf.variable_scope(scopename, reuse=reuse):
                cell = ConvGRUCell(shape, self.gf_dim * 2, self.convlstm_kernel)
                outputs, state = tf.nn.dynamic_rnn(cell, seq, dtype=seq.dtype)
            scopename = 'back_convRnn_' + str(unit_index)
            if self.use_gt: outputs = self.conditional_use_gt(outputs, unit_index, original_seq)
            with tf.variable_scope(scopename, reuse=reuse):
                cell = ConvGRUCell(shape, self.gf_dim * 2, self.convlstm_kernel)
                seq = tf.reverse(outputs, axis=[1])
                outputs, state = tf.nn.dynamic_rnn(cell, seq, dtype=seq.dtype)
            return tf.reverse(outputs, axis=[1])
        elif self.model == 'bi_parallel_progressive':
            return self.convRnn_seq_op(unit_index, seq, reuse=tf.AUTO_REUSE)
        else:
            scopename = ('for_convRnn_' if unit_index % 2 == 0 else 'back_convRnn_') + str(unit_index)
            with tf.variable_scope(scopename, reuse=reuse):
                cell = ConvGRUCell(shape, self.gf_dim * 2, self.convlstm_kernel)
                seq = seq if unit_index % 2 == 0 else tf.reverse(seq, axis=[1])
                outputs, state = tf.nn.dynamic_rnn(cell, seq, dtype=seq.dtype)
            return outputs if unit_index % 2 == 0 else tf.reverse(outputs, axis=[1])

    def assemble_hidden(self, seq, i, reuse=tf.AUTO_REUSE, train=True):
        seq_shape = seq.get_shape().as_list()
        concat_seq = tf.expand_dims(seq[:, 0, :, :, :], 1)
        # pass the concat and reserve previous
        for j in range(1, (2**(i+1)) * self.B + self.K):
            if j % 2 == 0:
                concat_seq = tf.concat([concat_seq, tf.expand_dims(seq[:, j // 2, :, :, :], 1)], axis= 1)
            else:
                concat_seq = tf.concat([concat_seq,
                    tf.expand_dims(self.concat_conv(
                        seq[:, j // 2, :, :, :], seq[:, j // 2 + 1, :, :, :], reuse=reuse, train=train),1)], axis=1)
        return concat_seq

    def forward_progressively(self, for_seq, is_gen):
        reuse = tf.AUTO_REUSE
        original_seq = None
        for_seq, res = self.pooling_feature_enc(for_seq, reuse=reuse, is_progressive=True)
        print "for_seq.get_shape().as_list()", for_seq.get_shape().as_list()
        print "len(res)", len(res)
        if self.use_gt: original_seq = tf.identity(for_seq)
        seq_shape = for_seq.get_shape().as_list()
        i = tf.constant(0, dtype=tf.int32)
        self.counter = 0
        # final_hidden = tf.zeros([seq_shape[0], self.length, seq_shape[2], seq_shape[3], seq_shape[4]])
        for i in range(self.convlstm_layer_num):

            for_seq = self.convRnn_progressive_op(i, original_seq,
                self.assemble_hidden(for_seq, i, reuse=reuse, train=is_gen), reuse=reuse)
            if self.use_gt:
                for_seq = self.conditional_use_gt(for_seq, i, original_seq)
            if i == 0:
                final_hidden = self.conv_layer_weight[i] * self.fill_create_frames(for_seq, res, i, reuse=reuse, train=is_gen)
            else:
                final_hidden += self.conv_layer_weight[i] * self.fill_create_frames(for_seq, res, i, reuse=reuse, train=is_gen)
            print "inner for loop", i, for_seq.get_shape().as_list()
        return self.get_create_frames(final_hidden)

    def get_frame(self, t, index, seq, res, reuse=False, train=True):
        res_t = 0 if self.res_mode == "mix" or self.res_mode == "one" else (t // (self.K+self.T))
        return tf.reshape(self.dec_cnn_depool(seq[:, index, :, :, :],
                            res[res_t], reuse=reuse, train=train),
                          [self.batch_size, self.image_size[0], self.image_size[1], 1, self.c_dim])

    def get_target_frames(self, hidden):
        frames = tf.expand_dims(hidden[:, :, :, 0, :], axis=3)
        frames_normal_weight = tf.expand_dims(hidden[:, :, :, 0, :], axis=3)
        # one = tf.constant(1, dtype=tf.int32)
        section = (self.K + self.T) // (2 ** (self.conv_layer_index + 1))
        for t in xrange(1, self.length):
            frames_normal_weight = tf.cond(tf.equal(0, t % section),
                             lambda: tf.concat((frames_normal_weight,
                                    tf.expand_dims(hidden[:, :, :, t, :], axis=3)), axis=3),
                             lambda: frames_normal_weight)
            frames = tf.cond(tf.equal(0, t % section),
                             lambda: tf.concat((frames, tf.cond(tf.equal(0, t % (2*section)),
                                    lambda: 1.0, lambda: self.loss_reduce_weight) *
                                    tf.expand_dims(hidden[:, :, :, t, :], axis=3)), axis=3),
                             lambda: frames)
        return frames, frames_normal_weight

    def get_create_frames(self, hidden):
        frames = tf.expand_dims(hidden[:, :, :, 0, :], axis=3)
        frames_normal_weight = tf.identity(frames)
        section = (self.K + self.T) // (2 ** (self.conv_layer_index + 1))
        for t in xrange(1, self.length):
            frames_normal_weight = tf.cond(tf.equal(0, t % section),
                             lambda: tf.concat([frames_normal_weight, tf.cond(tf.equal(0, t % (2*section)),
                                    lambda: 1.0, lambda: 1 / self.loss_reduce_weight) *
                                    tf.expand_dims(hidden[:, :, :, t, :], axis=3)], axis=3),
                             lambda: frames_normal_weight)
            frames = tf.cond(tf.equal(0, t % section),
                             lambda: tf.concat((frames, tf.expand_dims(hidden[:, :, :, t, :], axis=3)), axis=3),
                             lambda: frames)
        return frames, frames_normal_weight


    def fill_create_frames(self, hidden, res, i, reuse=False, train=True):
        seq_shape = hidden.get_shape().as_list()
        frames = self.get_frame(0, 0, hidden, res, reuse=reuse, train=train)
        # frames_normal_weight = tf.identity(frames)
        # section = (self.K + self.T) // (2 ** (self.conv_layer_index + 1))
        blank = tf.zeros([seq_shape[0], self.image_size[0], self.image_size[1], 1, self.c_dim])
        section = (self.K + self.T) // (2 ** (i + 1))
        for t in xrange(1, self.length):
            frames = tf.concat([frames, self.get_frame(t, t / section, hidden, res, reuse=reuse, train=train)], axis=3) \
                    if t % section == 0 else tf.concat([frames, blank], axis=3)
        return frames

    def concat_conv(self, pre_h, pos_h, reuse=tf.AUTO_REUSE, train=True):
        # 16 * 16 * (64 * 2)
        combine = tf.concat((pre_h, pos_h), axis = 3)
        conv1 = lrelu(batch_norm(conv2d(combine, output_dim=self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1, name='con_conv1', reuse=reuse),"concat_bn1", reuse=reuse, train=train))

        conv2 = lrelu(batch_norm(conv2d(conv1, output_dim=self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1, name='con_conv2', reuse=reuse),"concat_bn2", reuse=reuse, train=train))
        return conv2

    def pooling_feature_enc(self, seq, reuse = False, is_progressive=False):
        feature=None
        res_frame = []
        length = seq.get_shape().as_list()[1]
        blank_holder = None
        for k in range(length):
            if k % (self.K + self.T) < self.K:
                res = []
                ref_frame = tf.reshape(seq[:, k, :, :, :],[-1, self.image_size[0], self.image_size[1],self.c_dim])

                conv1_1 = lrelu(conv2d(ref_frame,
                       output_dim=self.gf_dim, k_h=3, k_w=3, d_h=2, d_w=2, name='fea_conv1_1', reuse=reuse))

                conv1_2 = lrelu(conv2d(conv1_1,
                       output_dim=self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='fea_conv1_2', reuse=reuse))
                # conv1_2 64*64*32
                res.append(conv1_2)
                if self.res_mode == "mix":
                    if len(res_frame) > 1:
                        res_frame[0][0] += conv1_2 / (self.B +self.K)
                    else:
                        res[0] = res[0] / (self.B + self.K)
                conv2_1 = lrelu(conv2d(conv1_2,
                       output_dim=self.gf_dim * 2, k_h=3, k_w=3, d_h=2, d_w=2, name='fea_conv2_1', reuse=reuse))
                conv2_2 = lrelu(conv2d(conv2_1,
                       output_dim=self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1, name='fea_conv2_2', reuse=reuse))
                # conv2_2 32*32*64
                res.append(conv2_2)
                if self.res_mode == "mix":
                    if len(res_frame) > 1:
                        res_frame[0][1] += conv2_2 / (self.B +self.K)
                    else:
                        res[1] = res[1] / (self.B +self.K)
                conv3_1 = lrelu(conv2d(conv2_2,
                       output_dim=self.gf_dim * 2, k_h=3, k_w=3, d_h=2, d_w=2, name='fea_conv3_1', reuse=reuse))
                conv3_2 = lrelu(conv2d(conv3_1,
                       output_dim=self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1, name='fea_conv3_2', reuse=reuse))
                # conv3 16*16*64
                # res.append(conv3_1)
                # if self.res_mode == "mix":
                #     if len(res_frame) > 1:
                #         res_frame[0][2] += conv3_1 / (self.B +self.K)
                #     else:
                #         res[2] = res[2] / (self.B +self.K)
                if feature is None:
                    feature=tf.reshape(conv3_2,[-1, 1, conv3_2.get_shape().as_list()[1],
                                                conv3_2.get_shape().as_list()[2], conv3_2.get_shape().as_list()[3]])
                    blank_holder = tf.zeros_like(feature)
                else:
                    feature=tf.concat([feature,
                                       tf.reshape(conv3_2,[-1, 1, conv3_2.get_shape().as_list()[1],
                                            conv3_2.get_shape().as_list()[2], conv3_2.get_shape().as_list()[3]])
                                       ], 1)
                res_frame.append(res)
                reuse = True
            elif not is_progressive:
                if feature is None:
                    raise Exception("feature shouldn't be NAN")
                else:
                    feature = tf.concat([feature, blank_holder], 1)
        if self.debug:
            print "pooling_feature_enc, feature:{}".format(feature.get_shape())
        return feature, res_frame

    # def encode_cnn(self, seq, reuse=False):
    def dec_cnn(self, comb, res, reuse=False, train=True):
        stride = int((self.image_size[0] / float(comb.get_shape().as_list()[1]))**(1/3.))
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
        deconv3 = relu(batch_norm(deconv2d(deconv2, output_shape=shape3, k_h=4, k_w=4,
                       d_h=stride, d_w=stride, name='dec_deconv3', reuse=reuse), "dec_bn3",reuse=reuse,train=train))
        shapeout = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]
        xtp1 = tanh(deconv2d(deconv3, output_shape=shapeout, k_h=4, k_w=4,
                             d_h=1, d_w=1, name='dec_deconv_out', reuse=reuse))
        if self.debug:
            print "dec_cnn,xtp1:{}".format(xtp1.get_shape())
        return xtp1

    def dec_cnn_depool(self, comb, res, reuse=True, train=True):
        stride = int((self.image_size[0] / float(comb.get_shape().as_list()[1]))**(1/2.))
        if self.debug:
            print "dec_cnn_pool comb:{}".format(comb.get_shape().as_list()[1])
            print "dec_cnn_pool stride:{}".format(stride)
        shapeout3 = [self.batch_size, comb.get_shape().as_list()[1] * stride,
                  comb.get_shape().as_list()[2] * stride, self.gf_dim * 2]
        # comb n * 16 * 16 * 64
        depool3 = FixedUnPooling(comb, [2, 2])
        # comb n * 32 * 32 * cat(64, 64)
        deconv3_2 = lrelu(batch_norm(deconv2d(self.conditional_combine(depool3,res[1]), output_shape=shapeout3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv3_2', reuse=reuse), "dec_bn32", reuse=reuse, train=train))
        deconv3_1 = lrelu(batch_norm(deconv2d(deconv3_2, output_shape=shapeout3, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv3_1', reuse=reuse), "dec_bn31", reuse=reuse, train=train))
        shapeout2 = [self.batch_size, deconv3_1.get_shape().as_list()[1] * stride,
                     deconv3_1.get_shape().as_list()[2] * stride, self.gf_dim]
        depool2 = FixedUnPooling(deconv3_1, [2, 2])
        # comb n * 64 * 64 * cat(64, 64)
        deconv2_2 = lrelu(batch_norm(deconv2d(self.conditional_combine(depool2,res[0]), output_shape=shapeout2, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv2_2', reuse=reuse), "dec_bn22", reuse=reuse, train=train))
        deconv2_1 = lrelu(batch_norm(deconv2d(deconv2_2, output_shape=shapeout2, k_h=3, k_w=3,
                                  d_h=1, d_w=1, name='dec_deconv2_1', reuse=reuse), "dec_bn21", reuse=reuse, train=train))
        shapeout1 = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]
        depool1 = FixedUnPooling(deconv2_1, [2, 2])
        # comb n * 128 * 128 * cat(32, 32)

        xtp1 = tanh(deconv2d(depool1, output_shape=shapeout1, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='dec_deconv1_1', reuse=reuse))
        if self.debug:
            print "dec_cnn,xtp1:{}".format(xtp1.get_shape())
        return xtp1

    def conditional_combine(self, layer, res):
        return tf.concat([layer,res], axis = 3) if self.Unet else layer

    def discriminator(self, image, train=True):
        h0 = lrelu(conv2d(image, self.df_dim, name='dis_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, self.df_dim, name='dis_h1_conv'),
                              "bn1", train=train))
        h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 2, name='dis_h2_conv'),
                              "bn2",train=train))
        h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 4, name='dis_h3_conv'),
                              "bn3",train=train))
        h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

        return tf.nn.sigmoid(h), h

    # def discriminator(self, image, train=True):
    #     h0 = lrelu(conv2d(image, self.df_dim, name='dis_h0_conv'))
    #     h1 = lrelu(conv2d(h0, self.df_dim * 2, name='dis_h1_conv'))
    #     h2 = lrelu(conv2d(h1, self.df_dim * 4, name='dis_h2_conv'))
    #     h3 = lrelu(conv2d(h2, self.df_dim * 8, name='dis_h3_conv'))
    #     h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')
    #
    #     return tf.nn.sigmoid(h), h

    def save(self, sess, checkpoint_dir, step):
        model_name = "bi_conv_lstm.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, model_name=None):
        print(" [*] Reading checkpoints... ")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None: model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            print("     Loaded model: " + str(model_name))
            return True, model_name
        else:
            return False, None
