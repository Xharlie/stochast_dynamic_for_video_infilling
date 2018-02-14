import os
import tensorflow as tf
import sys
from dr_utils import *


class UnetDCGAN(object):
    # opt = lapp[[
    # --learningRate(default 0.002)             learning rate
    # --beta1            (default 0.5)               momentum term for adam
    # -b, --batchSize    (default 100)               batch size
    # -g, --gpu          (default 0)                 gpu to use
    # --save             (default 'logs/')           base directory to save logs
    # --name             (default 'default')         checkpoint name
    # --dataRoot         (default '/path/to/data/')  data root directory
    # --optimizer        (default 'adam')            optimizer to train with
    # --nEpochs          (default 300)               max training epochs
    # --seed             (default 1)                 random seed
    # --epochSize        (default 50000)             number of samples per epoch
    # --contentDim       (default 64)               dimensionality of noise space
    # --poseDim          (default 16)               dimensionality of noise space
    # --imageSize        (default 64)                size of image
    # --dataset          (default moving_mnist)              dataset
    # --movingDigits     (default 1)
    # --cropSize         (default 227)               size of crop ( for kitti only)
    # --maxStep          (default 12)
    # --nShare           (default 1)                 number of frame to use for content encoding
    # --advWeight        (default 0)                 weight on adversarial scene discriminator loss
    # --normalize if set normalize pose and action vectors to have unit norm
    # --model            (default 'dcgan')
    # --unet             (default 'dcgan')
    # --nThreads         (default 0)                 number of dataloading threads
    # --dataPool         (default 200)
    # --dataWarmup       (default 10)
    # --beta             (default 0.0001)
    # ]]
    def __init__(self, args):
        self.args=args
        self.ngf = 64
        self.nc = args.color_channel_num

    def build_model(self):
        self.x_same_1 = tf.placeholder(tf.float32,
               [None, self.args.image_size_h, self.args.image_size_w, self.nc * self.args.nShare], name='x_same_1')
        self.x_same_2 = tf.placeholder(tf.float32,
               [None, self.args.image_size_h, self.args.image_size_w, self.nc * self.args.nShare], name='x_same_2')
        self.x_diff_3 = tf.placeholder(tf.float32,
               [None, self.args.image_size_h, self.args.image_size_w, self.nc * self.args.nShare], name='x_diff_3')
        self.same_label = tf.placeholder(tf.float32, [None, 1], name='same_label')
        self.diff_label = tf.placeholder(tf.float32, [None, 1], name='diff_label')
        self.is_gen = tf.placeholder(tf.bool, name="is_gen")
        self.is_dis = tf.placeholder(tf.bool, name="is_dis")
        self.x_single_1 = self.x_same_1[:,:,:,:self.nc]
        self.x_single_2 = self.x_same_2[:,:,:,:self.nc]
        self.x_single_3 = self.x_diff_3[:,:,:,:self.nc]
        if self.args.pose_net == "dcgan":
            hp1 = self.pose_encoder(self.x_single_1, train=self.is_gen, reuse=False)
            hp2 = self.pose_encoder(self.x_single_2, train=self.is_gen, reuse=True)
            hp3 = self.pose_encoder(self.x_single_3, train=self.is_gen, reuse=True)
        else:
            hp1 = self.pose_encoderRes18(self.x_single_1, train=self.is_gen, reuse=False)
            hp2 = self.pose_encoderRes18(self.x_single_2, train=self.is_gen, reuse=True)
            hp3 = self.pose_encoderRes18(self.x_single_3, train=self.is_gen, reuse=True)

        if self.args.unet == "dcgan":
            self.pred2, hc1 = self.makeUnet(self.x_same_1, hp2, reuse=False, train=self.is_gen)
            self.pred1, hc2 = self.makeUnet(self.x_same_2, hp1, reuse=True, train=self.is_gen)
            self.pred3_1, hc3 = self.makeUnet(self.x_diff_3, hp1, reuse=True, train=self.is_gen)
        else:
            self.pred2, hc1 = self.makeUnetVGG(self.x_same_1, hp2, reuse=False, train=self.is_gen)
            self.pred1, hc2 = self.makeUnetVGG(self.x_same_2, hp1, reuse=True, train=self.is_gen)
            self.pred3_1, hc3 = self.makeUnetVGG(self.x_diff_3, hp1, reuse=True, train=self.is_gen)

        same_prob, same_logit = self.scene_discriminator(hp1, hp2, reuse=False, train = self.is_dis)
        diff_prob, diff_logit = self.scene_discriminator(hp1, hp3, reuse=True, train = self.is_dis)
        label = tf.concat([self.same_label, self.diff_label], 0)
        logit = tf.concat([same_logit, diff_logit], 0)
    #     -------------- min hc1-hc2 --------------
        self.hc_same_loss = tf.reduce_mean(tf.square(hc1 - hc2))
    #     -------------- maximize entropy of scene discriminator output -------------
        self.entropy_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    #     ---------------- reconstruction loss -----------------------
        self.rec_loss = tf.reduce_mean(tf.square(self.x_single_1 - self.pred1) + tf.square(self.x_single_2 - self.pred2))
        self.gen_loss = self.rec_loss + self.hc_same_loss + self.args.beta * self.entropy_loss
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if not var.name.startswith('C_')]
        self.d_vars = [var for var in self.t_vars if var.name.startswith('C_')]
        num_param = 0.0
        for var in self.g_vars:
            num_param += int(np.prod(var.get_shape()))
        print("Number of parameters: %d" % num_param)

        self.hc_same_loss_sum = tf.summary.scalar("hc_same_loss", self.hc_same_loss)
        self.entropy_loss_max_sum = tf.summary.scalar("entropy_loss_max", self.entropy_loss)
        self.entropy_loss_min_sum = tf.summary.scalar("entropy_loss_min", self.entropy_loss)
        self.rec_loss_sum = tf.summary.scalar("rec_loss", self.rec_loss)
        self.gen_merge_sum = tf.summary.merge([self.hc_same_loss_sum ,
                       self.entropy_loss_max_sum, self.rec_loss_sum])

        self.train_seq_img = tf.placeholder(tf.float32, [None,
                             self.args.image_size_h * 5,
                             self.args.image_size_w * (self.args.batch_size // 5),
                             self.nc])
        self.test_seq_img = tf.placeholder(tf.float32, [None,
                             self.args.image_size_h * 5,
                             self.args.image_size_w * (self.args.batch_size // 5),
                             self.nc])

        self.train_seq_img_summary = tf.summary.image('train_seq_img', self.train_seq_img)
        self.test_seq_img_summary = tf.summary.image('test_seq_img', self.test_seq_img)
        self.summary_merge_plot_img = tf.summary.merge([
            self.train_seq_img_summary, self.test_seq_img_summary])

        self.saver = tf.train.Saver(max_to_keep=10)


    def pose_encoder(self, x, train=True, reuse=False):
        h1 = self.dcgan_layer(x, self.ngf, "EP_1", train=train, reuse=reuse)
        h2 = self.dcgan_layer(h1, self.ngf * 2, "EP_2", train=train, reuse=reuse)
        h3 = self.dcgan_layer(h2, self.ngf * 4, "EP_3", train=train, reuse=reuse)
        h4 = self.dcgan_layer(h3, self.ngf * 8, "EP_4", train=train, reuse=reuse)
        h5 = self.dcgan_layer(h4, self.ngf * 8, "EP_5", train=train, reuse=reuse)
        h6 = tanh(batch_norm(conv2d(h5, self.args.poseDim, name="EP_6", padding='VALID', reuse=reuse),
                             "EP_6" + "_bn", reuse=reuse, train=train))
        if self.args.normalize:
            return tf.nn.l2_normalize(h6, 3)
        return h6

    def pose_encoderRes18(self, x, train=True, reuse=False):
        # 128->64
        h1 = lrelu(batch_norm(conv2d(x, self.ngf, k_h=5, k_w=5, name="EP_res1", reuse=reuse),
                              "EP_res1" + "_bn", train=train, reuse=reuse))
        # 64->32
        mp_1 = MaxPooling(h1, 3 , 2, "SAME")
        l = resnet_backbone(mp_1,[2,2,2,2],train=train)
        h2 = tanh(batch_norm(conv2d(l, self.args.poseDim, name="EP_res2", padding='VALID', reuse=reuse),
                             "EP_res2" + "_bn", reuse=reuse, train=train))
        if self.args.normalize:
            return tf.nn.l2_normalize(h2, 3)
        return h2

    def scene_discriminator(self, first, second, reuse=False, train=True):
        nf = 100
        pair = tf.concat((first, second),3)
        h1 = relu(linear(tf.squeeze(pair,axis=[1,2]), nf, "C_1", reuse=reuse))
        h2 = relu(linear(h1, nf, "C_2", reuse=reuse))
        h3 = linear(h2, 1, "C_3", reuse=reuse)
        return tf.nn.sigmoid(h3), h3

    def makeUnet(self, x, pose, reuse=False, train=True):
        enc1 = self.dcgan_layer(x, self.ngf, "EC_e1", train=train, reuse=reuse)
        enc2 = self.dcgan_layer(enc1, self.ngf * 2, "EC_e2", train=train, reuse=reuse)
        enc3 = self.dcgan_layer(enc2, self.ngf * 4, "EC_e3", train=train, reuse=reuse)
        enc4 = self.dcgan_layer(enc3, self.ngf * 8, "EC_e4", train=train, reuse=reuse)
        enc5 = self.dcgan_layer(enc4, self.ngf * 8, "EC_e5", train=train, reuse=reuse)
        enc6 = tanh(batch_norm(conv2d(
            enc5, self.args.contentDim, name="EC_e6", padding='VALID', reuse=reuse), "EC_e6" + "_bn", train=train, reuse=reuse))
        join = tf.concat((enc6, pose), 3)
        dec1 = lrelu(batch_norm(deconv2d(join,
                  output_shape = [tf.shape(join)[0], 4, 4, self.ngf * 8],
                  k_h = 4, k_w = 4, d_h=4, d_w=4, name = "EC_d1", reuse=reuse), "EC_d1" + "_bn", train=train, reuse=reuse))
        dec2 = self.dec_layer(tf.concat((dec1, enc5), 3), self.ngf * 8, "EC_d2", train=train, reuse=reuse)
        dec3 = self.dec_layer(tf.concat((dec2, enc4), 3), self.ngf * 4, "EC_d3", train=train, reuse=reuse)
        dec4 = self.dec_layer(tf.concat((dec3, enc3), 3), self.ngf * 2, "EC_d4", train=train, reuse=reuse)
        dec5 = self.dec_layer(tf.concat((dec4, enc2), 3), self.ngf, "EC_d5", train=train, reuse=reuse)
        dec6 = tf.nn.tanh(deconv2d(tf.concat((dec5, enc1), 3),
                    output_shape = [tf.shape(x)[0], x.get_shape().as_list()[1], x.get_shape().as_list()[2], self.nc],
                    k_h=4, k_w=4,d_h=2, d_w=2, name="EC_d6", reuse=reuse))
        return dec6, enc6

    def vgg_layer(self, input, nout, name, train=True, reuse=False):
        return lrelu(batch_norm(conv2d(input, nout, k_h=3, k_w=3, d_h=1, d_w=1,name=name, reuse=reuse),
                                name + "_bn", train=train, reuse=reuse))

    def makeUnetVGG(self, x, pose, reuse=False, train=True):
    #     128 -> 64
        c1_1 = self.vgg_layer(x, self.ngf, "EC_e11", train=train, reuse=reuse)
        c1_2 = self.vgg_layer(c1_1, self.ngf, "EC_e12", train=train, reuse=reuse)
        mp_1 = MaxPooling(c1_2, 2,2)
    #     64 -> 32
        c2_1 = self.vgg_layer(mp_1, self.ngf*2, "EC_e21", train=train, reuse=reuse)
        c2_2 = self.vgg_layer(c2_1, self.ngf*2, "EC_e22", train=train, reuse=reuse)
        mp_2 = MaxPooling(c2_2, 2,2)
    #     32 -> 16
        c3_1 = self.vgg_layer(mp_2, self.ngf * 4, "EC_e31", train=train, reuse=reuse)
        c3_2 = self.vgg_layer(c3_1, self.ngf * 4, "EC_e32", train=train, reuse=reuse)
        c3_3 = self.vgg_layer(c3_2, self.ngf * 4, "EC_e33", train=train, reuse=reuse)
        mp_3 = MaxPooling(c3_3, 2,2)
    #     16 -> 8
        c4_1 = self.vgg_layer(mp_3, self.ngf * 8, "EC_e41", train=train, reuse=reuse)
        c4_2 = self.vgg_layer(c4_1, self.ngf * 8, "EC_e42", train=train, reuse=reuse)
        c4_3 = self.vgg_layer(c4_2, self.ngf * 8, "EC_e43", train=train, reuse=reuse)
        mp_4 = MaxPooling(c4_3, 2, 2)
    #     8 -> 4
        c5_1 = self.vgg_layer(mp_4, self.ngf * 8, "EC_e51", train=train, reuse=reuse)
        c5_2 = self.vgg_layer(c5_1, self.ngf * 8, "EC_e52", train=train, reuse=reuse)
        c5_3 = self.vgg_layer(c5_2, self.ngf * 8, "EC_e53", train=train, reuse=reuse)
        mp_5 = MaxPooling(c5_3, [2,2], [2,2], "SAME")
    #     4 -> 1
        c6 = tanh(batch_norm(conv2d(mp_5, self.args.contentDim, k_h=4, k_w=4, d_h=1, d_w=1,
                    name="EC_e6", padding='VALID', reuse=reuse), "EC_e6" + "_bn", train=train, reuse=reuse))
        join = tf.concat((c6, pose), 3)
        comb = lrelu(batch_norm(deconv2d(join, output_shape=[tf.shape(join)[0], 4, 4, self.ngf * 8]
                                         , k_h=4, k_w=4, d_h=4, d_w=4, name="EC_d0",
                     reuse=reuse), "EC_d0" + "_bn", train=train, reuse=reuse))
    #   4 -> 8
        up_1 = FixedUnPooling(comb, [2, 2])
        d1_1 = self.vgg_layer(tf.concat((up_1, c5_3), 3), self.ngf * 8, "EC_d11", train=train, reuse=reuse)
        d1_2 = self.vgg_layer(d1_1, self.ngf * 8, "EC_d12", train=train, reuse=reuse)
        d1_3 = self.vgg_layer(d1_2, self.ngf * 8, "EC_d13", train=train, reuse=reuse)
    #     8 -> 16
        up_2 = FixedUnPooling(d1_3, [2, 2])
        d2_1 = self.vgg_layer(tf.concat((up_2, c4_3), 3), self.ngf * 8, "EC_d21", train=train, reuse=reuse)
        d2_2 = self.vgg_layer(d2_1, self.ngf * 8, "EC_d22", train=train, reuse=reuse)
        d2_3 = self.vgg_layer(d2_2, self.ngf * 4, "EC_d23", train=train, reuse=reuse)
    #     16 -> 32
        up_3 = FixedUnPooling(d2_3, [2, 2])
        d3_1 = self.vgg_layer(tf.concat((up_3, c3_3), 3), self.ngf * 4, "EC_d31", train=train, reuse=reuse)
        d3_2 = self.vgg_layer(d3_1, self.ngf * 4, "EC_d32", train=train, reuse=reuse)
        d3_3 = self.vgg_layer(d3_2, self.ngf * 2, "EC_d33", train=train, reuse=reuse)
    #     32 -> 64
        up_4 = FixedUnPooling(d3_3, [2, 2])
        d4_1 = self.vgg_layer(tf.concat((up_4, c2_2), 3), self.ngf * 2, "EC_d41", train=train, reuse=reuse)
        d4_2 = self.vgg_layer(d4_1, self.ngf, "EC_d42", train=train, reuse=reuse)
    #     64 -> 128
        up_5 = FixedUnPooling(d4_2, [2, 2])
        d5_1 = self.vgg_layer(tf.concat((up_5, c1_2), 3), self.ngf * 2, "EC_d51", train=train, reuse=reuse)
        d5_2 = conv2d(d5_1, self.nc, k_h=3, k_w=3, d_h=1, d_w=1, name="EC_d52", reuse=reuse)
        return tf.nn.tanh(d5_2), c6

    def dcgan_layer(self, input, nout, name, train=True, reuse=False):
        return lrelu(batch_norm(conv2d(input, nout, name=name, reuse=reuse), name + "_bn", train=train, reuse=reuse))

    def dec_layer(self, input, nout, name, train=True, reuse=False):
        output_shape = [tf.shape(input)[0], input.get_shape().as_list()[1] * 2,
                        input.get_shape().as_list()[2] * 2, nout]
        return lrelu(batch_norm(deconv2d(input, output_shape = output_shape, k_h=4, k_w=4,
                      d_h=2, d_w=2, name=name, reuse=reuse), name + "_bn",train=train, reuse=reuse))



    def save(self, sess, checkpoint_dir, step):
        model_name = "drnet.model"

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


