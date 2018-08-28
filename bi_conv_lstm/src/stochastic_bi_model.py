import os
import tensorflow as tf

import tensorflow.contrib
from convlstm_cell import *
from ops import *
from utils import *
from dr_bi_utils import *


class stochastic_bi_net(object):
    def __init__(self, image_size, batch_size=32, c_dim=1, K=1, T=3, B=5, debug = False, pixel_loss="l2",
                convlstm_kernel = [3, 3], mode="bi_sto", space_aware = True, cell_type='lstm', z_channel=24,
                 normalize = True, weight = 3, res_type="first", negative_noise = True, res_ref = True, pic_norm = True):
        self.pixel_loss = pixel_loss
        self.convlstm_kernel = convlstm_kernel
        self.batch_size = batch_size
        self.image_size = image_size
        self.pattern_dim = 128
        self.z_dimension = z_channel
        self.ngf = 64
        self.c_dim = c_dim
        self.K = K
        self.T = T
        self.B = B
        self.length = K+T+1
        self.ref_shape = [batch_size, B+1, self.image_size[0], self.image_size[1], c_dim]
        self.inf_shape = [batch_size, K+T+1, self.image_size[0], self.image_size[1], c_dim]
        self.debug = debug
        self.mode = mode
        self.space_aware = space_aware
        self.cell_type = cell_type
        self.normalize = normalize
        self.weight = weight
        self.res_type = res_type
        self.negative_noise = negative_noise
        self.res_ref = res_ref
        self.pic_norm = pic_norm
        self.build_model()

    def build_model(self):
        self.ref_seq = tf.placeholder(tf.float32, self.ref_shape, name='ref_seq')
        self.inf_seq = tf.placeholder(tf.float32, self.inf_shape, name='inf_seq')
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.p_loss_percentage = tf.placeholder(tf.float32, name="is_train")
        if self.pic_norm:
            batch_min = tf.reduce_min(self.ref_seq, axis=[1,2,3,4])
            batch_max = tf.reduce_max(self.ref_seq, axis=[1,2,3,4])
            print "batch_min.get_shape().as_list()", batch_min.get_shape().as_list()
            self.batch_mean  = tf.reshape(tf.div(tf.add(batch_min, batch_max), 2.0), [self.batch_size,1,1,1,1])
            self.batch_scale = tf.reshape(tf.div(tf.subtract(batch_max, batch_min), 2.0), [self.batch_size,1,1,1,1])
            self.ref_seq_norm = tf.div(
                tf.subtract(self.ref_seq, self.batch_mean), self.batch_scale)
            self.inf_seq_norm = tf.div(
                tf.subtract(self.inf_seq, self.batch_mean), self.batch_scale)
            print "self.ref_seq.get_shape().as_list()", self.ref_seq.get_shape().as_list()
        else:
            self.ref_seq_norm = self.ref_seq
            self.inf_seq_norm = self.inf_seq
        # batch * 4 * 4 * lh
        for_state, back_state = extract_pattern(self, self.ref_seq_norm, self.is_train)
        # inf_hidden_seq batch * (0:K+T+1) * 4 * 4 * pattern_dim
        self.inf_hidden_seq, self.res_list, self.res_hidden = serial_pose_encoder(
            self, self.inf_seq_norm, self.ref_seq_norm, self.is_train, reuse=tf.AUTO_REUSE)
        with tf.variable_scope('frame_prediction_lstm', reuse=tf.AUTO_REUSE):
            # for_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.poseDim, state_is_tuple=True)
            shape = for_state.get_shape().as_list()[1:3]
            # shape = 4 * 4
            if self.cell_type == "lstm":
                self.prediction_cell = ConvLSTMCell(shape, self.pattern_dim + self.z_dimension, self.convlstm_kernel)
            else:
                self.prediction_cell = ConvGRUCell(shape, self.pattern_dim + self.z_dimension, self.convlstm_kernel)
        # 1:K+T(exclusive)
        prior_mus, prior_sigmas, prior_zs, predicted_frames \
            = self.prior_gaussian_lstm(self.inf_hidden_seq, for_state, back_state, is_train = self.is_train)
        # from 1 to K+T exclusive
        post_mus, post_sigmas, post_zs = post_gaussian_lstm(self, self.inf_hidden_seq)

        recon_size = 3
        self.G_post, _ = post_gen_graph(self, post_zs)
        self.G = tf.reshape(tf.concat(predicted_frames, axis=1),
                            [self.batch_size, self.K + self.T - 1, self.image_size[0], self.image_size[1],
                             self.c_dim])
        self.last_frame = tf.reshape( tf.zeros_like(self.G[:,-1,...]), [self.batch_size, 1, self.image_size[0], self.image_size[1],
                        self.c_dim])
        if self.weight != 0:
            self.mask, self.mask_binary = create_mask(self.inf_seq_norm,
                      weight=self.weight, negative_noise=self.negative_noise, c_dim=self.c_dim)
        else:
            self.mask = tf.ones_like(self.G_post)
            self.mask_binary = tf.zeros_like(self.G_post)
        # pixel level loss
        if self.pixel_loss == "l2":
            self.L_train_p = self.p_loss_percentage * tf.reduce_mean(
                                tf.multiply(self.mask,
                                            tf.square(self.G_post - self.inf_seq_norm[:, 1:-1, ...])
                                            )
                            ) + \
                            (1.0 - self.p_loss_percentage) * tf.reduce_mean(
                                tf.multiply(self.mask,
                                            tf.square(self.G - self.inf_seq_norm[:, 1:-1, ...])
                                            )
                            )
        else:
            self.L_train_p = self.p_loss_percentage * tf.reduce_mean(
                                tf.multiply(self.mask,
                                            tf.abs(self.G_post - self.inf_seq_norm[:, 1:-1, ...])
                                            )
                            ) + \
                            (1.0 - self.p_loss_percentage) * tf.reduce_mean(
                                tf.multiply(self.mask,
                                            tf.abs(self.G - self.inf_seq_norm[:, 1:-1, ...])
                                            )
                            )

        if self.pic_norm:
            self.G_real = tf.add(tf.multiply(self.G, self.batch_scale), self.batch_mean)
            self.G_post_real = tf.add(tf.multiply(self.G_post, self.batch_scale), self.batch_mean)
        else:
            self.G_real = self.G
            self.G_post_real = self.G_post
        self.L_train_p_l1_diff = tf.reduce_mean(
                        tf.abs(self.G_post_real - self.inf_seq[:, 1:-1, ...])
        )
        self.L_test_p_l1_diff = tf.reduce_mean(
            tf.abs(self.G_real - self.inf_seq[:, 1:-1, ...])
        )
        # reappear loss
        # self.L_reappear = tf.Variable(0)
        # prehidden loss
        # self.L_prehidden_loss = tf.Variable(0)
        # recon loss
        # self.L_hidden_recon_loss = tf.Variable(0)
        # KL loss
        self.L_train_kl = 0
        self.L_train_kl_exlusive = 0
        for k in range(len(prior_mus)):
            self.L_train_kl += kl_loss(self.batch_size, post_mus[k],
                               post_sigmas[k], prior_mus[k], prior_sigmas[k])
            self.L_train_kl_exlusive += kl_loss(self.batch_size, prior_mus[k], prior_sigmas[k],
                                    post_mus[k], post_sigmas[k])
        self.L_train_kl = self.L_train_kl / len(prior_mus)

        self.L_train_p_sum = tf.summary.scalar("L_train_p", self.L_train_p)
        self.L_test_p_sum = tf.summary.scalar("L_test_p", self.L_train_p)
        self.L_train_p_l1_diff_sum = tf.summary.scalar("L_train_p_l1_diff", self.L_train_p_l1_diff)
        self.L_testTrain_p_l1_diff_sum = tf.summary.scalar("L_testTrain_p_l1_diff", self.L_train_p_l1_diff)
        self.L_trainTest_p_l1_diff_sum = tf.summary.scalar("L_trainTest_p_l1_diff", self.L_test_p_l1_diff)
        self.L_test_p_l1_diff_sum = tf.summary.scalar("L_test_p_l1_diff", self.L_test_p_l1_diff)
        self.L_train_kl_sum = tf.summary.scalar("L_train_kl", self.L_train_kl)
        # self.L_train_appear_sum = tf.summary.scalar("L_train_reappear", self.L_reappear)
        # self.L_test_appear_sum = tf.summary.scalar("L_test_reappear", self.L_reappear)
        # self.L_train_prehidden_sum = tf.summary.scalar("L_train_prehidden", self.L_prehidden_loss)
        # self.L_test_prehidden_sum = tf.summary.scalar("L_test_prehidden", self.L_prehidden_loss)
        # self.L_train_hidden_recon_sum = tf.summary.scalar("L_train_hidden_recon", self.L_hidden_recon_loss)
        # self.L_test_hidden_recon_sum = tf.summary.scalar("L_test_hidden_recon", self.L_hidden_recon_loss)

        self.mean_batch_psnr_train_post = tf.placeholder(tf.float32, name='mean_batch_psnr_train_post')
        self.mean_batch_psnr_train = tf.placeholder(tf.float32, name='mean_batch_psnr_train')
        self.mean_batch_psnr_test_post = tf.placeholder(tf.float32, name='mean_batch_psnr_test_post')
        self.mean_batch_psnr_test = tf.placeholder(tf.float32, name='mean_batch_psnr_test')

        self.mean_batch_ssim_train_post = tf.placeholder(tf.float32, name='mean_batch_ssim_train_post')
        self.mean_batch_ssim_train = tf.placeholder(tf.float32, name='mean_batch_ssim_train')
        self.mean_batch_ssim_test_post = tf.placeholder(tf.float32, name='mean_batch_ssim_test_post')
        self.mean_batch_ssim_test = tf.placeholder(tf.float32, name='mean_batch_ssim_test')

        self.mean_batch_psnr_train_post_sum = tf.summary.scalar("mean_batch_psnr_train_post_sum", self.mean_batch_psnr_train_post)
        self.mean_batch_psnr_train_sum = tf.summary.scalar("mean_batch_psnr_train_sum", self.mean_batch_psnr_train)
        self.mean_batch_psnr_test_post_sum = tf.summary.scalar("mean_batch_psnr_test_post_sum", self.mean_batch_psnr_test_post)
        self.mean_batch_psnr_test_sum = tf.summary.scalar("mean_batch_psnr_test_sum", self.mean_batch_psnr_test)

        self.mean_batch_ssim_train_post_sum = tf.summary.scalar("mean_batch_ssim_train_post_sum", self.mean_batch_ssim_train_post)
        self.mean_batch_ssim_train_sum = tf.summary.scalar("mean_batch_ssim_train_sum", self.mean_batch_ssim_train)
        self.mean_batch_ssim_test_post_sum = tf.summary.scalar("mean_batch_ssim_test_post_sum", self.mean_batch_ssim_test_post)
        self.mean_batch_ssim_test_sum = tf.summary.scalar("mean_batch_ssim_test_sum", self.mean_batch_ssim_test)

        self.train_vars = tf.trainable_variables()
        de_vars = filter(lambda x: x.name.startswith('DE'), tf.trainable_variables())
        dup_de_vars = filter(lambda x: x.name.startswith('du_DE'), tf.trainable_variables())
        self.trainable_variables = filter(lambda x: not x.name.startswith('du_DE'), tf.trainable_variables())
        print "len(de_vars), len(dup_de_vars)", len(de_vars), len(dup_de_vars)
        if len(de_vars) == len(dup_de_vars):
            for ix, var in enumerate(de_vars):
                copy_from_en = tf.assign(dup_de_vars[ix], var.value())
                tf.add_to_collection("update_dup", copy_from_en)

        # self.g_vars_rnn = [var for var in self.t_vars if 'DIS' not in var.name and 'EC' not in var.name and 'EP' not in var.name]
        # self.g_vars_all = [var for var in self.t_vars if 'DIS' not in var.name]
        # self.g_vars_en_de = [var for var in self.t_vars if 'DIS' not in var.name and ('EC' in var.name or 'EP' in var.name)]
        # self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
        # self.EP_vars = [var for var in self.t_vars if 'EP' in var.name]
        # self.EC_vars = [var for var in self.t_vars if 'EC' in var.name]
        num_param = 0.0
        self.test_seq_img = tf.placeholder(tf.uint8, [None, self.image_size[0] * 4 * recon_size,
                                                        self.image_size[1] * (self.B + self.T + self.K),
                                                        self.c_dim])
        self.train_seq_img = tf.placeholder(tf.uint8, [None, self.image_size[0] * 4 * recon_size,
                                                         self.image_size[1] * (self.B + self.T + self.K),
                                                         self.c_dim])
        self.test_seq_img_summary = tf.summary.image('test_seq_img', self.test_seq_img)
        self.train_seq_img_summary = tf.summary.image('train_seq_img', self.train_seq_img)
        self.summary_merge_seq_img = tf.summary.merge([
            self.train_seq_img_summary, self.test_seq_img_summary])
        self.summary_merge_metrics = tf.summary.merge([
            self.mean_batch_psnr_train_post_sum, self.mean_batch_psnr_train_sum,
            self.mean_batch_psnr_test_post_sum, self.mean_batch_psnr_test_sum,
            self.mean_batch_ssim_train_post_sum, self.mean_batch_ssim_train_sum,
            self.mean_batch_ssim_test_post_sum, self.mean_batch_ssim_test_sum
        ])
        for var in self.train_vars:
            num_param += int(np.prod(var.get_shape()))
        print("Number of parameters: %d" % num_param)
        # print(self.train_vars)
        self.saver = tf.train.Saver(max_to_keep=10)

    #
    # def test_graph(self, predicted_frames):
    #     G = tf.reshape(tf.concat(predicted_frames, axis=1),
    #                    [self.batch_size, self.K + self.T - 1, self.image_size[0], self.image_size[1],
    #                     self.c_dim])
    #     return G

    def prior_gaussian_lstm(self, inf_hidden_seq, for_state, back_state, is_train = True, reuse=tf.AUTO_REUSE):
        # 0:K+T+1, length = K+T+1
        length = inf_hidden_seq.get_shape().as_list()[1]
        shape = inf_hidden_seq.get_shape().as_list()[2:4]
        place_zeros = tf.zeros_like(inf_hidden_seq[:,1:-1,...])
        seq = tf.concat([tf.expand_dims(inf_hidden_seq[:,0,...], axis=1),
                         place_zeros, tf.expand_dims(inf_hidden_seq[:,-1,...], axis=1)], axis=1)
        for i in range(1):
            with tf.variable_scope('for_prior_gaussian_mean_lstm_' + str(i), reuse=reuse):
                # for_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.poseDim, state_is_tuple=True)
                if self.cell_type == "lstm":
                    for_cell = ConvLSTMCell(shape, self.pattern_dim, self.convlstm_kernel)
                else:
                    for_cell = ConvGRUCell(shape, self.pattern_dim, self.convlstm_kernel)
            with tf.variable_scope('back_prior_gaussian_mean_lstm_' + str(i), reuse=reuse):
                # back_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.poseDim, state_is_tuple=True)
                if self.cell_type == "lstm":
                    back_cell = ConvLSTMCell(shape, self.pattern_dim, self.convlstm_kernel)
                else:
                    back_cell = ConvGRUCell(shape, self.pattern_dim, self.convlstm_kernel)
            with tf.variable_scope('bidirection_prior_gaussian_mean_rnn_' + str(i), reuse=reuse):
                if i == 0:
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(for_cell, back_cell, seq,
                        initial_state_fw=tf.nn.rnn_cell.LSTMStateTuple(for_state, tf.zeros_like(for_state)),
                        initial_state_bw=tf.nn.rnn_cell.LSTMStateTuple(back_state, tf.zeros_like(back_state)), dtype=seq.dtype)
                else:
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(for_cell, back_cell, seq, dtype=seq.dtype)
            if i == 0:
                seq = conv_time(tf.concat(outputs, 4), self.pattern_dim, 'gaussian_mean_conv_'+str(i))
            else:
                seq = seq + conv_time(tf.concat(outputs, 4), self.pattern_dim, 'gaussian_mean_conv_'+str(i))
        if self.normalize:
            seq = tf.nn.l2_normalize(seq, 4)

        with tf.variable_scope('gaussian_std_lstm', reuse=reuse):
            dimension = self.pattern_dim
            if self.cell_type == "lstm":
                prediction_gaussian_std_cell = ConvLSTMCell(shape, self.pattern_dim, self.convlstm_kernel)
            else:
                prediction_gaussian_std_cell = ConvGRUCell(shape, self.pattern_dim, self.convlstm_kernel)
            state = prediction_gaussian_std_cell.zero_state(self.batch_size, tf.float32)

        def training_inf(seq, prediction_gaussian_std_cell, get_spatial_mean,
                         get_spatial_std, get_spatial_z, z_dimension, inf_hidden_seq, state,
                         prediction_state, frame_prediction, decoder, res_list, res_hidden):
            prior_mus = []
            prior_sigmas = []
            prior_zs = []
            predicted_frames = []
            for j in range(1, length - 1):
                h_minus_one = inf_hidden_seq[:, j - 1, ...]
                # dimension pattern_dim + lh_dim
                with tf.variable_scope('prediction_gaussian_lstm', reuse=reuse):
                    # out is batch * 4 * 4 * pattern_dim
                    _, state = prediction_gaussian_std_cell(h_minus_one, state)
                    out, _ = prediction_gaussian_std_cell(seq[:,j,...], state)
                with tf.variable_scope('prior_z', reuse=reuse):
                    mu = get_spatial_mean(out, z_dimension)
                    sigma = get_spatial_std(out, z_dimension)
                    z = get_spatial_z(mu,sigma)
                prior_mus.append(mu)
                prior_sigmas.append(sigma)
                prior_zs.append(z)
                predicted_frame_hidden, prediction_state = \
                    frame_prediction(h_minus_one, z, prediction_state, is_train=False)
                print("res shape:", res_list)
                predicted_frame = decoder(self, predicted_frame_hidden, res_list, res_hidden, is_train=False, reuse=reuse)
                predicted_frames.append(predicted_frame)
            return prior_mus, prior_sigmas, prior_zs, predicted_frames

        def test_inf(seq, prediction_gaussian_std_cell, get_spatial_mean,
                    get_spatial_std, get_spatial_z, z_dimension, inf_hidden_seq, state,
                    prediction_state, frame_prediction, decoder, res_list, res_hidden):
            prior_mus = []
            prior_sigmas = []
            prior_zs = []
            predicted_frames = []
            h_minus_one = None
            for j in range(1, length - 1):
                if j==1:
                    h_minus_one = inf_hidden_seq[:,j-1,...]
                with tf.variable_scope('prediction_gaussian_lstm', reuse=reuse):
                    _, state = prediction_gaussian_std_cell(h_minus_one, state)
                    out, _ = prediction_gaussian_std_cell(seq[:,j,...], state)
                with tf.variable_scope('prior_z', reuse=reuse):
                    mu = get_spatial_mean(out, z_dimension)
                    sigma = get_spatial_std(out, z_dimension)
                    z = get_spatial_z(mu, sigma)
                prior_mus.append(mu)
                prior_sigmas.append(sigma)
                prior_zs.append(z)
                predicted_frame_hidden, prediction_state =\
                    frame_prediction(h_minus_one, z, prediction_state, is_train = False)
                print("res shape:", self.res_list)
                predicted_frame = decoder(self,
                    predicted_frame_hidden, res_list, res_hidden, is_train=False, reuse=reuse)
                h_minus_one, _ = pose_encoder(self, predicted_frame, is_train=False, reuse=reuse)
                predicted_frames.append(tf.expand_dims(predicted_frame,axis=1))
            return prior_mus, prior_sigmas, prior_zs, predicted_frames
        # seq is hidden embedding for seq
        prediction_state = self.prediction_cell.zero_state(self.batch_size, tf.float32)
        return tf.cond(self.is_train,
                       lambda: training_inf(seq, prediction_gaussian_std_cell,
                       get_spatial_mean, get_spatial_std, get_spatial_z, self.z_dimension, inf_hidden_seq,
                       state, prediction_state, self.frame_prediction, decoder, self.res_list, self.res_hidden),
                       lambda: test_inf(seq, prediction_gaussian_std_cell,
                       get_spatial_mean, get_spatial_std, get_spatial_z, self.z_dimension, inf_hidden_seq,
                       state, prediction_state, self.frame_prediction, decoder, self.res_list, self.res_hidden))

    def frame_prediction(self, h_minus_one, z, prediction_state, is_train = True):
        with tf.variable_scope('frame_prediction_lstm', reuse=tf.AUTO_REUSE):
            out, prediction_state = self.prediction_cell(
                tf.concat([h_minus_one, z], axis = 3), prediction_state)
        return out, prediction_state

    # def du_decoder(self, hidden, res, is_train = False, reuse = tf.AUTO_REUSE):
    #     ratio = 4
    #     if self.space_aware:
    #         ratio = 1
    #     dec1 = self.dec_layer(hidden, self.ngf * 4, "du_DE_d1_1", ratio= ratio, is_train = is_train, reuse=reuse)
    #     if res[0].get_shape().as_list()[1] == 64:
    #         dec1 = self.dec_layer(tf.concat((dec1, res[4]), 3), self.ngf * 4, "du_DE_d1_2", is_train=is_train, reuse=reuse)
    #     dec2 = self.dec_layer(tf.concat((dec1, res[3]), 3), self.ngf * 4, "du_DE_d2", is_train=is_train, reuse=reuse)
    #     dec3 = self.dec_layer(tf.concat((dec2, res[2]), 3), self.ngf * 2, "du_DE_d3", is_train=is_train, reuse=reuse)
    #     dec4 = self.dec_layer(tf.concat((dec3, res[1]), 3), self.ngf, "du_DE_d4", is_train=is_train, reuse=reuse)
    #     # dec5 = self.dec_layer(tf.concat((dec4, res[0]), 3), self.ngf, "EC_d5", is_train=is_train, reuse=reuse)
    #     dec5 = tf.nn.tanh(deconv2d(tf.concat((dec4, res[0]), 3),
    #                                output_shape=[tf.shape(res[0])[0], self.image_size[0], self.image_size[1],
    #                                              self.c_dim],
    #                                k_h=4, k_w=4, d_h=2, d_w=2, name="du_DE_d5", reuse=reuse))
    #     shape = dec5.get_shape().as_list()
    #     print "du_decoder h5 shape", shape
    #     return dec5

    # def conditional_combine(self, layer, res):
    #     return tf.concat([layer,res], axis = 3) if self.Unet else layer

    # def discriminator(self, image, is_train=True):
    #     h0 = lrelu(conv2d(image, self.df_dim, name='dis_h0_conv'))
    #     h1 = lrelu(batch_norm(conv2d(h0, self.df_dim, name='dis_h1_conv'),
    #                           "bn1", is_train=is_train))
    #     h2 = lrelu(batch_norm(conv2d(h1, self.df_dim * 2, name='dis_h2_conv'),
    #                           "bn2",is_train=is_train))
    #     h3 = lrelu(batch_norm(conv2d(h2, self.df_dim * 4, name='dis_h3_conv'),
    #                           "bn3",is_train=is_train))
    #     h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')
    #
    #     return tf.nn.sigmoid(h), h

    # def discriminator(self, image, is_train=True):
    #     h0 = lrelu(conv2d(image, self.df_dim, name='dis_h0_conv'))
    #     h1 = lrelu(conv2d(h0, self.df_dim * 2, name='dis_h1_conv'))
    #     h2 = lrelu(conv2d(h1, self.df_dim * 4, name='dis_h2_conv'))
    #     h3 = lrelu(conv2d(h2, self.df_dim * 8, name='dis_h3_conv'))
    #     h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')
    #
    #     return tf.nn.sigmoid(h), h


