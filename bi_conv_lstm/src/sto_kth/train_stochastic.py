import sys
import time
import tensorflow as tf
import glob

from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser


sys.path.append("../")
sys.path.append("../../../data/stochastic_dataset_script/")
from stochastic_bi_model import stochastic_bi_net
from stochastic_learned_prior import stochastic_learned_prior
from deter_flexible import deter_flexible
import data_util
import metrics
import ops
from utils import *

parser = ArgumentParser()
parser.add_argument("--lr", type=float, dest="lr",
                    default=0.002, help="Base Learning Rate")
parser.add_argument("--batch_size", type=int, dest="batch_size",
                    default=8, help="Mini-batch size")
parser.add_argument("--beta1", type=float, dest="beta1",
                    default=0.5, help="beta1 of adams")
parser.add_argument("--alpha", type=float, dest="alpha",
                    default=0.0001, help="inclusion KL weight")
parser.add_argument("--gamma", type=float, dest="gamma",
                    default=0.0, help="exclusion KL weight")
parser.add_argument("--beta", type=float, dest="beta",
                    default=0.4, help="posterior and inference weight")
parser.add_argument("--z_channel", type=int, dest="z_channel",
                    default=24, help="Image loss weight")
parser.add_argument("--decay_step", type=float, dest="decay_step",
                    default=40000)
parser.add_argument("--decay_rate", type=float, dest="decay_rate",
                    default=0.9)
parser.add_argument("--image_size_h", type=int, dest="image_size_h",
                    default=64, help="Mini-batch size")
parser.add_argument("--image_size_w", type=int, dest="image_size_w",
                    default=64, help="Mini-batch size")
parser.add_argument("--K", type=int, dest="K",
                    default=1, help="Number of frames of gt per block")
parser.add_argument("--T", type=int, dest="T",
                    default=3, help="Number of frames synthesized per block")
parser.add_argument("--B", type=int, dest="B",
                    default=5, help="number of blocks")
parser.add_argument("--num_iter", type=int, dest="num_iter",
                    default=120001, help="Number of iterations")
parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", default=0,
                    help="GPU device id")
parser.add_argument("--cpu", action="store_true", dest="cpu", help="use cpu only")
parser.add_argument("--no_normalized", action="store_true", dest="no_normalized", help="no_normalized")
parser.add_argument("--mask_weight", type=int, dest="mask_weight", default=3, help="mask_weight")
parser.add_argument("--start_percentage", type=float, dest="start_percentage",
                    default=1.0, help="start_percentage")
parser.add_argument("--load_pretrain", action="store_true", dest="load_pretrain", help="load_pretrain")
parser.add_argument("--color_channel_num", type=int, dest="color_channel_num",
                    default=1, help="number of color channels")
parser.add_argument("--debug", action="store_true", dest="debug", help="debug mode")
parser.add_argument("--no_store", action="store_true", dest="no_store", help="no_store")
parser.add_argument("--no_res_ref", action="store_true", dest="no_res_ref", help="no_res_ref")
parser.add_argument("--no_negative_noise", action="store_true", dest="no_negative_noise", help="no_negative_noise")
parser.add_argument("--no_space_aware", action="store_true", dest="no_space_aware", help="space_aware")
parser.add_argument("--no_pic_norm", action="store_true", dest="no_pic_norm", help="no_pic_norm")
parser.add_argument("--iters", type=int, dest="iters", default=0)
parser.add_argument("--mode", type=str, dest="mode",
                        default="bi_sto", help="number frames to be sent in discriminator")
parser.add_argument("--cell_type", type=str, dest="cell_type",
                        default="gru", help="lstm or gru")
parser.add_argument("--pretrain_model", type=str, dest="pretrain_model",
                    default="", help="")
parser.add_argument("--pixel_loss", type=str, dest="pixel_loss",
                    default="l1", help="")
parser.add_argument("--res_type", type=str, dest="res_type",
                    default="avg", help="")
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')

args = parser.parse_args()
space_aware = not args.no_space_aware

def image_clipping(img):
    img = inverse_transform(img) * 255
    img = np.clip(img, 0, 255, out=img)
    return img.astype(np.uint8)

def main():

    train_vids, test_vids = data_util.load_dataset(args)
    iters = args.iters
    prefix = ("sto"
              + "_h=" + str(args.image_size_h)
              + "_w=" + str(args.image_size_w)
              + "_K=" + str(args.K)
              + "_T=" + str(args.T)
              + "_B=" + str(args.B)
              + "_batch_size=" + str(args.batch_size)
              + "_beta1=" + str(args.beta1)
              + "_alpha=" + str(args.alpha)
              + "_gamma=" + str(args.gamma)
              + "_lr=" + str(args.lr)
              + "_mode=" + str(args.mode)
              + "_space_aware=" + str(space_aware)
              + "_z_channel=" + str(args.z_channel)
              + "_p_loss=" + str(args.pixel_loss)
              + "_cell_type=" + str(args.cell_type)
              + "_norm=" + str(not args.no_normalized)
              + "_mask_w=" + str(args.mask_weight)
              + "_res_type=" + str(args.res_type)
              + "_neg_noise=" + str(not args.no_negative_noise)
              + "_res_ref=" + str(not args.no_res_ref)
              + "_pic_norm=" + str(not args.no_pic_norm)
              + "_start_perc=" + str(args.start_percentage)
    )

    print("\n" + prefix + "\n")
    checkpoint_dir = "../../models/stochastic/" + args.dataset + '/' + prefix + "/"
    samples_dir = "../../samples/stochastic/" + args.dataset + '/' + prefix + "/"
    summary_dir = "../../logs/stochastic/" + args.dataset + '/' + prefix + "/"

    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)
    #     save synthesized frame sample
    if not exists(samples_dir):
        makedirs(samples_dir)
    if not exists(summary_dir):
        makedirs(summary_dir)

    device_string = ""
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device_string = "/cpu:0"
    elif args.gpu:
        device_string = "/gpu:%d" % args.gpu[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu[0])

    with tf.device(device_string):
        if args.mode == "bi_sto":
            model = stochastic_bi_net([args.image_size_h, args.image_size_w], batch_size = args.batch_size,
                       c_dim = args.color_channel_num, K=args.K, T=args.T, B=args.B, debug = False,
                       pixel_loss = args.pixel_loss, convlstm_kernel = [3, 3], mode = args.mode,
                       space_aware = space_aware, cell_type=args.cell_type, z_channel = args.z_channel,
                       normalize = not args.no_normalized, weight=args.mask_weight, res_type=args.res_type,
                       negative_noise = not args.no_negative_noise, res_ref = not args.no_res_ref,
                       pic_norm = not args.no_pic_norm)
        elif args.mode == "learned_prior":
            model = stochastic_learned_prior([args.image_size_h, args.image_size_w], batch_size = args.batch_size,
                       c_dim = args.color_channel_num, K=args.K, T=args.T, B=args.B, debug = False,
                       pixel_loss = args.pixel_loss, convlstm_kernel = [3, 3], mode = args.mode,
                       space_aware = space_aware, cell_type=args.cell_type, z_channel = args.z_channel,
                       normalize = not args.no_normalized, weight=args.mask_weight, res_type=args.res_type,
                       negative_noise = not args.no_negative_noise, res_ref = not args.no_res_ref,
                       pic_norm = not args.no_pic_norm)
        elif args.mode == "deter_flexible":
            model = deter_flexible([args.image_size_h, args.image_size_w], batch_size = args.batch_size,
                       c_dim = args.color_channel_num, K=args.K, T=args.T, B=args.B, debug = False,
                       pixel_loss = args.pixel_loss, convlstm_kernel = [3, 3], mode = args.mode,
                       space_aware = space_aware, cell_type=args.cell_type, z_channel = args.z_channel,
                       normalize = not args.no_normalized, weight=args.mask_weight, res_type=args.res_type,
                       negative_noise = not args.no_negative_noise, res_ref = not args.no_res_ref,
                       pic_norm = not args.no_pic_norm)
        global_step = tf.Variable(0, trainable=False)
        global_rate = tf.train.exponential_decay(args.lr, global_step,
                     args.decay_step, args.decay_rate, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            g_full = model.L_train_p + args.alpha * model.L_train_kl
            if args.gamma != 0:
                g_full += args.gamma * model.L_train_kl_exlusive
            g_optim = tf.train.AdamOptimizer(global_rate, beta1=args.beta1).minimize(
                g_full, var_list=model.trainable_variables, global_step=global_step
            )
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        if args.load_pretrain:
            if ops.load(model, sess, checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        train_sum = tf.summary.merge([model.L_train_p_sum, model.L_train_kl_sum, model.L_train_p_l1_diff_sum,
                          model.L_trainTest_p_l1_diff_sum])
        test_sum = tf.summary.merge([model.L_test_p_sum, model.L_test_p_l1_diff_sum,
                                     model.L_testTrain_p_l1_diff_sum])

        writer = tf.summary.FileWriter(summary_dir, sess.graph)
        start_time = time.time()
        full_train = True
        # if (len(args.pretrain_model) > 0):
        #     # Create a saver. include gen_vars and encoder_vars
        #     model.saver.restore(sess, args.pretrain_model)
        blank = None
        p_loss_percentage = 1.0
        flipable = False
        if args.dataset=="kth":
            flipable = True
        blank1 = None
        while iters <= args.num_iter:
            mini_batches = get_minibatches_idx(len(train_vids), args.batch_size, shuffle=True)
            for _, batchidx in mini_batches:
                if args.start_percentage == 0.0:
                    p_loss_percentage = 0.5
                else:
                    if iters >= (args.num_iter * args.start_percentage):
                        if iters < args.num_iter * (1 - args.start_percentage):
                            p_loss_percentage = 1 - 0.6 * (
                                    (1.0 * iters / args.num_iter - args.start_percentage)
                                    / (1.0 - 2 * args.start_percentage))
                        else:
                            p_loss_percentage = 0.4
                if iters > args.num_iter: break
                if len(batchidx) == args.batch_size:
                    sess.run(tf.get_collection('update_dup'))
                    # batch, time, height, width, color
                    ref_batch, inf_batch = load_stochastic_data_from_list(train_vids, batchidx,
                                args.image_size_h, args.image_size_w, args.K, args.T,
                                  args.B, flipable=flipable, channel=args.color_channel_num)
                    if args.debug:
                        print ref_batch.shape, inf_batch.shape
                    _, summary_str , L_train_p, L_train_kl\
                        = sess.run([g_optim, train_sum, model.L_train_p, model.L_train_kl],
                                    feed_dict={model.ref_seq: ref_batch,
                                               model.inf_seq: inf_batch,
                                               model.is_train: True,
                                               model.p_loss_percentage: p_loss_percentage})
                    if not args.no_store: writer.add_summary(summary_str, iters)
                    print(
                        "Iters: [%2d] time: %4.4f, L_train_p: %.8f, L_train_kl: %.8f"
                            % (iters, time.time() - start_time, L_train_p, L_train_kl)
                    )

                    if np.mod(iters, 2500) == 0:
                        print("validation at iters:", iters)
                        ref_batch_train, inf_batch_train = load_stochastic_data_from_list(train_vids,
                                              range(3, 3 + args.batch_size/2)+range(60, 60 + args.batch_size/2),
                                              args.image_size_h, args.image_size_w,
                                              args.K, args.T, args.B, flipable=flipable, channel=args.color_channel_num)

                        ref_batch_test, inf_batch_test = load_stochastic_data_from_list(test_vids,
                                              range(3, 3 + args.batch_size/2)+range(60, 60 + args.batch_size/2),
                                              args.image_size_h,
                                              args.image_size_w,
                                              args.K, args.T, args.B, flipable=flipable, channel=args.color_channel_num)
                        if blank1 is None:
                            blank1 = np.zeros_like(ref_batch_train[0, :args.B // 2 + 1, ...])
                            blank2 = np.zeros_like(ref_batch_train[0, args.B//2+1: , ...])
                        summary_test, L_test_p, L_test_kl, \
                        G_test, G_test_post, test_mask_binary, last_frame_test = sess.run(
                            [test_sum, model.L_train_p, model.L_train_kl, model.G_real,
                             model.G_post_real, model.mask_binary, model.last_frame],
                            feed_dict={model.ref_seq: ref_batch_test,
                                       model.inf_seq: inf_batch_test,
                                       model.is_train: False,
                                       model.p_loss_percentage: p_loss_percentage})

                        _, _, _, _, _, mean_batch_psnr_test_post, mean_batch_ssim_test_post\
                            = metrics.cal_seq(inf_batch_test[:, 1:-1, ...], G_test_post)
                        _, _, _, _, _, mean_batch_psnr_test, mean_batch_ssim_test \
                            = metrics.cal_seq(inf_batch_test[:, 1:-1, ...], G_test)

                        writer.add_summary(summary_test, iters)
                        print(
                            "Iters: [%2d] time: %4.4f, L_test_p: %.8f, L_test_kl: %.8f"
                                % (iters, time.time() - start_time, L_test_p, L_test_kl)
                        )
                        print("ref_batch_test.min, ref_batch_test.max", np.min(ref_batch_test), np.max(ref_batch_test))
                        print("mean_batch_psnr_test_post, mean_batch_ssim_test_post",
                              mean_batch_psnr_test_post, mean_batch_ssim_test_post)
                        print("mean_batch_psnr_test, mean_batch_ssim_test",
                              mean_batch_psnr_test, mean_batch_ssim_test)
                        print "test G_test.shape", G_test.shape
                        summary_train, L_train_p, L_train_kl, G_train, \
                        G_train_post, train_mask_binary, last_frame_train = sess.run(
                            [train_sum, model.L_train_p, model.L_train_kl, model.G_real, model.G_post_real,
                             model.mask_binary, model.last_frame],
                            feed_dict={model.ref_seq: ref_batch_train,
                                       model.inf_seq: inf_batch_train,
                                       model.is_train: True,
                                       model.p_loss_percentage: p_loss_percentage})

                        _, _, _, _, _, mean_batch_psnr_train_post, mean_batch_ssim_train_post \
                            = metrics.cal_seq(inf_batch_train[:, 1:-1, ...], G_train_post)
                        _, _, _, _, _, mean_batch_psnr_train, mean_batch_ssim_train \
                            = metrics.cal_seq(inf_batch_train[:, 1:-1, ...], G_train)
                        print("mean_batch_psnr_train_post, mean_batch_ssim_train_post",
                              mean_batch_psnr_train_post, mean_batch_ssim_train_post)
                        print("mean_batch_psnr_train, mean_batch_ssim_train",
                              mean_batch_psnr_train, mean_batch_ssim_train)
                        for i in [1, args.batch_size/2 ,args.batch_size - 1]:
                            sample_train = depth_to_width(np.concatenate(
                                (ref_batch_train[i,:args.B//2,...],
                                 inf_batch_train[i,...], ref_batch_train[i,args.B//2+2:,...]), axis=0))
                            gen_train_mask = depth_to_width(np.concatenate(
                                (blank1, train_mask_binary[i, ...], blank2),axis=0))
                            gen_train_post = depth_to_width(np.concatenate(
                                (blank1, G_train_post[i, ...], blank2), axis=0))
                            gen_train = depth_to_width(np.concatenate(
                                (blank1, G_train[i, ...], blank2),axis=0))
                            sample_test = depth_to_width(np.concatenate(
                                (ref_batch_test[i,:args.B//2,...],
                                 inf_batch_test[i,...], ref_batch_test[i,args.B//2+2:,...]),axis=0))
                            gen_test_mask = depth_to_width(np.concatenate(
                                (blank1, test_mask_binary[i, ...], blank2), axis=0))
                            gen_test_post = depth_to_width(np.concatenate(
                                (blank1, G_test_post[i, ...], blank2), axis=0))
                            gen_test = depth_to_width(np.concatenate(
                                (blank1, G_test[i, ...], blank2),axis=0))
                            if i == 1:
                                print sample_train.shape, gen_train.shape, sample_train.shape
                                sample_train_cat = np.concatenate((sample_train, gen_train_mask, gen_train_post, gen_train), axis=0)
                                sample_test_cat = np.concatenate((sample_test, gen_test_mask, gen_test_post, gen_test), axis=0)
                            else:
                                sample_train_cat = np.concatenate(
                                    (sample_train_cat, sample_train, gen_train_mask, gen_train_post, gen_train), axis=0)
                                sample_test_cat = np.concatenate(
                                    (sample_test_cat, sample_test, gen_test_mask, gen_test_post, gen_test), axis=0)
                        print("Saving sample at iter"), iters
                        img_summary = sess.run(model.summary_merge_seq_img, feed_dict={
                            model.train_seq_img: np.expand_dims(image_clipping(sample_train_cat), axis=0),
                            model.test_seq_img: np.expand_dims(image_clipping(sample_test_cat), axis=0)
                        })
                        metrics_summary = sess.run(
                            model.summary_merge_metrics, feed_dict={
                                model.mean_batch_psnr_test_post: mean_batch_psnr_test_post,
                                model.mean_batch_psnr_test: mean_batch_psnr_test,
                                model.mean_batch_psnr_train_post: mean_batch_psnr_train_post,
                                model.mean_batch_psnr_train: mean_batch_psnr_train,
                                model.mean_batch_ssim_test_post: mean_batch_ssim_test_post,
                                model.mean_batch_ssim_test: mean_batch_ssim_test,
                                model.mean_batch_ssim_train_post: mean_batch_ssim_train_post,
                                model.mean_batch_ssim_train: mean_batch_ssim_train
                            }
                        )
                        if not args.no_store:
                            writer.add_summary(img_summary, iters)
                            writer.add_summary(metrics_summary, iters)
                    if np.mod(iters, 10000) == 0 and iters != 0 and not args.no_store:
                        ops.save(model, sess, checkpoint_dir, iters)
                iters += 1
        print "finish Training"
        # model.save(sess, checkpoint_dir, iters)


if __name__ == "__main__":
    main()
