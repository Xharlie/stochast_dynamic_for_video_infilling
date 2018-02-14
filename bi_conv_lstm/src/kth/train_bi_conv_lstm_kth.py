import sys
import time
import tensorflow as tf
import glob

from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser

sys.path.append("../")
import load_save_data
from bi_conv_lstm_model import bi_convlstm_net
from utils import *

def conditional_addframes(seq, section, model, B, T, K):
    length = B*(K+T)+K
    full_seq = np.zeros((seq.shape[0], seq.shape[1], length, seq.shape[3]))
    if model != "bi_pass":
        for i in range(length):
            if i % section < K:
                full_seq[:, :, i, :] = seq[:,:,i // section,:]
        return full_seq
    else:
        return seq

def gradually_hot(total, num, iters, num_iter):
    # num_layers = int(convlstm_layer_num * (float(iters) / num_iter)) + 1
    feedin = np.zeros((total), dtype=np.float)
    if num == 1:
        feedin[0] = 1
    else:
        alpha = min((total * (float(iters + 1) / num_iter) - num + 1) * 2, 1)
        feedin[num - 2] = 1 - alpha
        feedin[num - 1] = alpha
    return feedin

def one_hot(total, num):
    feedin = np.zeros((total),dtype=np.float)
    feedin[num-1] = 1
    return feedin

def main(lr, batch_size, alpha, beta, image_size_h, image_size_w, K, T, B, convlstm_layer_num, num_iter, gpu, cpu,
         load_pretrain, tf_record_train_dir, tf_record_test_dir, color_channel_num, dec,no_store,pixel_loss, pretrain_model,
         dyn_enc_model, reference_mode, debug, print_train_instead, dis_length, model, Unet, no_d, fade_in, use_gt, res_mode):

    check_create_dir(tf_record_train_dir)
    check_create_dir(tf_record_test_dir)
    train_tf_record_files = glob.glob(tf_record_train_dir + '*tfrecords*')
    test_tf_record_files = glob.glob(tf_record_test_dir + '*tfrecords*')

    train_vids = []
    if len(train_tf_record_files) == 0:
        print "len(train_tf_record_files) == 0:"
        data_path = "../../../data/KTH/"
        f = open(data_path + "train_data_list_trimmed.txt", "r")
        trainfiles = f.readlines()
        train_vids = load_save_data.save_data2record(
            trainfiles, data_path, image_size_h, image_size_w, tf_record_train_dir, color_channel_num)
    else:
        train_vids = load_save_data.load_records(glob.glob(tf_record_train_dir + '*tfrecords*'))
    print "len(train_vids)", len(train_vids)
    test_vids = []
    if len(test_tf_record_files) == 0:
        data_path = "../../../data/KTH/"
        f = open(data_path + "test_data_list.txt", "r")
        testfiles = f.readlines()
        test_vids = load_save_data.save_data2record(
            testfiles, data_path, image_size_h, image_size_w, tf_record_test_dir, color_channel_num)
    else:
        test_vids = load_save_data.load_records(glob.glob(tf_record_test_dir + '*tfrecords*'))
    print "len(test_vids)", len(test_vids)
    margin = 0.3
    updateD = True
    updateG = True
    iters = 0
    prefix = ("KTH_convlstm"
              + "_image_size_h=" + str(image_size_h)
              + "_image_size_w=" + str(image_size_w)
              + "_K=" + str(K)
              + "_T=" + str(T)
              + "_B=" + str(B)
              + "_convlstm_layer_num=" + str(convlstm_layer_num)
              + "_dec=" + str(dec)
              + "_dis_length=" + str(dis_length)
              + "_batch_size=" + str(batch_size)
              + "_alpha=" + str(alpha)
              + "_beta=" + str(beta)
              + "_lr=" + str(lr)
              + "_model=" + str(model)
              + "_fade_in=" + str(fade_in)
              + "_no_d=" + str(no_d)
              + "_use_gt=" + str(use_gt)
              + "_res_mode=" + str(res_mode)
              + "_pixel_loss=" + str(pixel_loss)
              + "_Unet=" + str(Unet))

    print("\n" + prefix + "\n")
    checkpoint_dir = "../../models/KTH/" + prefix + "/"
    samples_dir = "../../samples/KTH/" + prefix + "/"
    summary_dir = "../../logs/KTH/" + prefix + "/"

    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)
    #     save synthesized frame sample
    if not exists(samples_dir):
        makedirs(samples_dir)
    if not exists(summary_dir):
        makedirs(summary_dir)

    device_string = ""
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device_string = "/cpu:0"
    elif gpu:
        device_string = "/gpu:%d" % gpu[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu[0])
    with tf.device(device_string):
        model = bi_convlstm_net(dec=dec, image_size=[image_size_h, image_size_w], c_dim=color_channel_num, dis_length=dis_length,
                                K=K, T=T, B=B, convlstm_layer_num=convlstm_layer_num, batch_size=batch_size, checkpoint_dir=checkpoint_dir,
                                debug=debug, reference_mode=reference_mode, model = model, Unet= Unet, use_gt = use_gt, res_mode=res_mode,
                                pixel_loss=pixel_loss)
        # for var in tf.trainable_variables():
        #     print var.name
        # raw_input('print trainable variables...')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
                model.d_loss, var_list=model.d_vars
            )

            g_loss = alpha * model.L_img
            if not no_d: g_loss += beta * model.L_GAN
            g_optim_full = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
                g_loss, var_list=model.g_vars
            )

            g_optim_second = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
                g_loss, var_list=model.g_2_vars
            )
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = False

    with tf.Session(config=config) as sess:

        tf.global_variables_initializer().run()

        if load_pretrain:
            if model.load(sess, checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        if no_d:
            g_sum = tf.summary.merge([model.L_p_sum,
                                  model.L_gdl_sum, model.L_img_sum])
        else:
            g_sum = tf.summary.merge([model.L_p_sum,
                                  model.L_gdl_sum, model.L_img_sum,
                                  model.L_GAN_sum])
        d_sum = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                                  model.d_loss_fake_sum])
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        counter = iters + 1
        start_time = time.time()
        full_train = True
        g_optim = g_optim_full

        if (len(pretrain_model) > 0):
            # Create a saver. include gen_vars and encoder_vars
            model.saver.restore(sess, args.pretrain_model)
            iters = 20000
            counter = 20001

        while iters < num_iter:
            mini_batches = get_minibatches_idx(len(train_vids), batch_size, shuffle=True)
            for _, batchidx in mini_batches:
                if iters >= num_iter: break
                if len(batchidx) == batch_size:

                    #     seq_batch is (batch,w,h,t,c)
                    seq_batch = load_data_from_list(train_vids, batchidx, image_size_h, image_size_w, K, T, B)
                    if debug:
                        print seq_batch.shape
                    #     seq_batch_tran is (batch,t,w,h,c)
                    seq_batch_tran = create_missing_frames(seq_batch.copy().transpose([0, 3, 1, 2, 4]), K, T)
                    forward_seq = seq_batch_tran
                    num_layers = int(convlstm_layer_num * (float(iters) / num_iter)) + 1
                    if fade_in:
                        conv_layer_weight = gradually_hot(convlstm_layer_num, num_layers, iters, num_iter)
                        if num_layers > 1: print conv_layer_weight
                    else:
                        conv_layer_weight = one_hot(convlstm_layer_num, num_layers)
                        if num_layers > 1 and (convlstm_layer_num * (float(iters + 1) / num_iter) - num_layers + 1) * 3 < 1:
                            g_optim = g_optim_second
                            full_train = False
                        else:
                            full_train = True
                            g_optim = g_optim_full

                    if num_layers  != (int(convlstm_layer_num * (float(iters - 1) / num_iter)) + 1) :
                        print "num_layers:", num_layers
                    if full_train and updateD and (not no_d):
                        _, summary_str = sess.run([d_optim, d_sum],
                                                  feed_dict={model.forward_seq: forward_seq,
                                                             model.target: seq_batch,
                                                             model.conv_layer_weight: conv_layer_weight,
                                                             model.conv_layer_index: num_layers - 1,
                                                             model.loss_reduce_weight: conv_layer_weight[num_layers - 1],
                                                             model.is_dis: True,
                                                             model.is_gen: True})
                        if not no_store: writer.add_summary(summary_str, counter)

                    if (not full_train) or (no_d or updateG):
                        _, summary_str,L_p, L_gdl,G  = sess.run([g_optim, g_sum, model.L_p, model.L_gdl, model.G],
                                                  feed_dict={model.forward_seq: forward_seq,
                                                             model.target: seq_batch,
                                                             model.conv_layer_weight: conv_layer_weight,
                                                             model.conv_layer_index: num_layers - 1,
                                                             model.loss_reduce_weight: conv_layer_weight[num_layers - 1],
                                                             model.is_dis: True,
                                                             model.is_gen: True})
                        if not no_store: writer.add_summary(summary_str, counter)
                        print "train shape", G.shape
                    if not no_d:
                        errD_fake, errD_real, errG, L_p, L_gdl = sess.run(
                            [model.d_loss_fake, model.d_loss_real, model.L_GAN, model.L_p, model.L_gdl],
                            feed_dict={
                                model.forward_seq: forward_seq,
                                model.target: seq_batch,
                                model.conv_layer_weight: conv_layer_weight,
                                model.conv_layer_index: num_layers - 1,
                                model.loss_reduce_weight: conv_layer_weight[num_layers - 1],
                                model.is_dis: False,
                                model.is_gen: False
                            })

                        # if errD_fake < margin or errD_real < margin:
                        #     updateD = False
                        # if errD_fake > (1. - margin) or errD_real > (1. - margin):
                        #     updateG = False
                        # if not updateD and not updateG:
                        #     updateD = True
                        #     updateG = True
                    # var_value = [v for v in tf.global_variables() if v.name == "DIS/bn1/moving_mean:0"][0]
                    # print "DIS/bn1/moving_mean:0:", var_value.eval()
                    if no_d:
                        print(
                                "Iters: [%2d] time: %4.4f, L_p: %.8f, L_gdl: %.8f"
                                % (iters, time.time() - start_time, L_p, L_gdl)
                                )
                    else:
                        print("Iters: [%2d] time: %4.4f, errD_fake: %.8f, errD_real: %.8f, L_GAN: %.8f, L_p: %.8f, L_gdl: %.8f"
                            % (iters, time.time() - start_time, errD_fake, errD_real, errG, L_p, L_gdl)
                        )

                    if np.mod(counter, 2500) == 0:
                        seq_batch_train = load_data_from_list(train_vids, range(80, 80 + batch_size), image_size_h, image_size_w,
                                                              K, T, B, flipable=False)
                        forward_seq_train = create_missing_frames(seq_batch_train.copy().transpose([0, 3, 1, 2, 4]), K, T)
                        seq_batch_test = load_data_from_list(test_vids, range(0, batch_size), image_size_h, image_size_w,
                                                             K, T, B, flipable=False)
                        forward_seq_test = create_missing_frames(seq_batch_test.copy().transpose([0, 3, 1, 2, 4]), K, T)
                        samples_train = sess.run([model.G],
                                                 feed_dict={model.forward_seq: forward_seq_train,
                                                            model.target: seq_batch_train,
                                                            model.conv_layer_weight: conv_layer_weight,
                                                            model.conv_layer_index: num_layers - 1,
                                                            model.loss_reduce_weight: conv_layer_weight[num_layers - 1],
                                                            model.is_dis: False,
                                                            model.is_gen: False})[0]
                        samples_test = sess.run([model.G],
                                                feed_dict={model.forward_seq: forward_seq_test,
                                                           model.target: seq_batch_test,
                                                           model.conv_layer_weight: conv_layer_weight,
                                                           model.conv_layer_index: num_layers - 1,
                                                           model.loss_reduce_weight: conv_layer_weight[num_layers - 1],
                                                           model.is_dis: False,
                                                           model.is_gen: False})[0]
                        forward_seq_test = forward_seq_test.transpose([0, 2, 3, 1, 4])
                        forward_seq_train = forward_seq_train.transpose([0, 2, 3, 1, 4])
                        print "test num_layers, samples_train.shape", num_layers, samples_train.shape
                        for i in range(3):
                            sample_train = depth_to_width(conditional_addframes(
                                samples_train[i], (K+T) // (2**num_layers), model, B, T, K).swapaxes(0, 2).swapaxes(1, 2))
                            sbatch_train = depth_to_width(seq_batch_train[i, :, :, :].swapaxes(0, 2).swapaxes(1, 2))
                            fbatch_train = depth_to_width(forward_seq_train[i, :, :, :].swapaxes(0, 2).swapaxes(1, 2))
                            sample_test = depth_to_width(conditional_addframes(
                                samples_test[i], (K+T) // (2**num_layers), model, B, T, K).swapaxes(0, 2).swapaxes(1, 2))
                            sbatch_test = depth_to_width(seq_batch_test[i, :, :, :].swapaxes(0, 2).swapaxes(1, 2))
                            fbatch_test = depth_to_width(forward_seq_test[i, :, :, :].swapaxes(0, 2).swapaxes(1, 2))
                            if i == 0:
                                print sbatch_train.shape, fbatch_train.shape, sample_train.shape
                                sample_train_cat = np.concatenate((sbatch_train, fbatch_train, sample_train), axis=0)
                                sample_test_cat = np.concatenate((sbatch_test, fbatch_test, sample_test), axis=0)
                            else:
                                sample_train_cat = np.concatenate(
                                    (sample_train_cat, sbatch_train, fbatch_train, sample_train), axis=0)
                                sample_test_cat = np.concatenate(
                                    (sample_test_cat, sbatch_test, fbatch_test, sample_test), axis=0)
                            # save_images(sample[:,:,:,::-1], [2, B*(K+T)+K],
                            #             samples_dir+"train_%s_%s.png" % (iters, i))
                        print("Saving sample at iter"), iters
                        summary = sess.run(model.summary_merge_seq_img, feed_dict={
                            model.train_seq_img: np.expand_dims(inverse_transform(sample_train_cat) * 255, axis=0),
                            model.test_seq_img: np.expand_dims(inverse_transform(sample_test_cat) * 255, axis=0)
                        })
                        if not no_store: writer.add_summary(summary, counter)

                    counter += 1

                    if np.mod(counter, 10000) == 0 and not no_store:
                        model.save(sess, checkpoint_dir, counter)

                    iters += 1
        model.save(sess, checkpoint_dir, counter)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, dest="lr",
                        default=0.0001, help="Base Learning Rate")
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=8, help="Mini-batch size")
    parser.add_argument("--alpha", type=float, dest="alpha",
                        default=1.0, help="Image loss weight")
    parser.add_argument("--beta", type=float, dest="beta",
                        default=0.02, help="GAN loss weight")
    parser.add_argument("--image_size_h", type=int, dest="image_size_h",
                        default=128, help="Mini-batch size")
    parser.add_argument("--image_size_w", type=int, dest="image_size_w",
                        default=128, help="Mini-batch size")
    parser.add_argument("--K", type=int, dest="K",
                        default=1, help="Number of frames of gt per block")
    parser.add_argument("--T", type=int, dest="T",
                        default=3, help="Number of frames synthesized per block")
    parser.add_argument("--B", type=int, dest="B",
                        default=5, help="number of blocks")
    parser.add_argument("--convlstm_layer_num", type=int, dest="convlstm_layer_num",
                        default=2, help="number of convlstm layers")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=100000, help="Number of iterations")
    parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", default=0,
                        help="GPU device id")
    parser.add_argument("--cpu", action="store_true", dest="cpu", help="use cpu only")
    parser.add_argument("--Unet", action="store_true", dest="Unet", help="use Unet")
    parser.add_argument("--load_pretrain", action="store_true", dest="load_pretrain", help="load_pretrain")
    parser.add_argument("--tf_record_train_dir", type=str, nargs="?", dest="tf_record_train_dir",
                        default="../../../tf_record/KTH/train/", help="tf_record train location")
    parser.add_argument("--tf_record_test_dir", type=str, nargs="?", dest="tf_record_test_dir",
                        default="../../../tf_record/KTH/test/", help="tf_record test location")
    parser.add_argument("--color_channel_num", type=int, dest="color_channel_num",
                        default=1, help="number of color channels")
    parser.add_argument("--dec", type=str, dest="dec", default="deconv",
                        help="deconv or depool")
    parser.add_argument("--dyn_enc_model", type=str, dest="dyn_enc_model", default="mix",
                        help="dynamic encoding model")
    parser.add_argument("--reference_mode", type=str, dest="reference_mode", default="two",
                        help="refer to how many frames in the end")
    parser.add_argument("--debug", action="store_true", dest="debug", help="debug mode")
    parser.add_argument("--no_d", action="store_true", dest="no_d", help="debug mode")
    parser.add_argument("--fade_in", action="store_true", dest="fade_in", help="fade_in")
    parser.add_argument("--use_gt", action="store_true", dest="use_gt", help="use_gt")
    parser.add_argument("--no_store", action="store_true", dest="no_store", help="no_store")
    # one, mix, or close(not recommend)
    parser.add_argument("--res_mode", type=str, default="mix", dest="res_mode", help="res_mode")
    parser.add_argument("--pixel_loss", type=str, default="l2", dest="pixel_loss", help="pixel_loss")
    parser.add_argument("--print_train_instead", action="store_true",
                        dest="print_train_instead", help="print_train_instead of test")
    parser.add_argument("--dis_length", type=int, dest="dis_length",
                        default=11, help="number frames to be sent in discriminator")
    # bi_serial_progressive, bi_parallel_progressive or bi_pass
    parser.add_argument("--model", type=str, dest="model",
                        default="bi_pass", help="number frames to be sent in discriminator")
    parser.add_argument("--pretrain_model", type=str, dest="pretrain_model",
                        default="", help="")
    args = parser.parse_args()
    main(**vars(args))
