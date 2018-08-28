import os
import cv2
import sys
import time
import ssim
import imageio

import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw
import glob

sys.path.append("../")
from bi_conv_lstm_model import bi_convlstm_net
from utils import *

def main(lr, batch_size, alpha, beta, image_size_h, image_size_w, K, T, B, convlstm_layer_num, num_iter, gpu, cpu,
         concatconv, quantitative, qualitative,
         load_pretrain, tf_record_train_dir, tf_record_test_dir, color_channel_num, dec, no_store, pixel_loss,
         pretrain_model, mode,
         dyn_enc_model, reference_mode, debug, print_train_instead, dis_length, Unet, no_d, fade_in, use_gt,
         res_mode, gif_per_vid):
    data_path = "../../../data/KTH/"
    f = open(data_path + "test_data_list.txt", "r")
    testfiles = f.readlines()
    c_dim = 1
    iters = 0

    best_model = None  # will pick last model
    prefix = ("KTH_convlstm"
              + "_image_size_h=" + str(image_size_h)
              + "_image_size_w=" + str(image_size_w)
              + "_K=" + str(K)
              + "_T=" + str(T)
              + "_B=" + str(B)
              + "_convlstm_layer_num=" + str(convlstm_layer_num)
              + "_dis_length=" + str(dis_length)
              + "_batch_size=" + str(batch_size)
              + "_alpha=" + str(alpha)
              + "_beta=" + str(beta)
              + "_lr=" + str(lr)
              + "_mode=" + str(mode)
              + "_fade_in=" + str(fade_in)
              + "_no_d=" + str(no_d)
              + "_use_gt=" + str(use_gt)
              + "_res_mode=" + str(res_mode)
              + "_pixel_loss=" + str(pixel_loss)
              + "_concatconv=" + str(concatconv)
              + "_Unet=" + str(Unet))
    checkpoint_dir = "../../models/KTH/" + prefix + "/"
    device_string = ""
    if cpu:
        device_string = "/cpu:0"
    elif gpu:
        device_string = "/gpu:%d" % gpu[0]
    with tf.device(device_string):
        # test batch size has to be 1

        model = bi_convlstm_net(dec=dec, image_size=[image_size_h, image_size_w], c_dim=color_channel_num,
                                dis_length=dis_length, K=K, T=T, B=B, convlstm_layer_num=convlstm_layer_num, batch_size=1,
                                checkpoint_dir=checkpoint_dir, debug=debug, reference_mode=reference_mode, mode=mode,
                                Unet=Unet, use_gt=False, res_mode=res_mode, pixel_loss=pixel_loss, concatconv=concatconv)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        tf.global_variables_initializer().run()
        print checkpoint_dir
        loaded, model_name = model.load(sess, checkpoint_dir, best_model)

        if loaded:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed... exitting")
            return

        quant_dir = "../../results/quantitative/KTH/" + prefix + "/"
        save_path = quant_dir + "results_model=" + model_name + ".npz"
        if not exists(quant_dir):
            makedirs(quant_dir)

        vid_names = []
        length = B * (K + T) + K
        psnr_err = np.zeros((0, length))
        ssim_err = np.zeros((0, length))
        for i in xrange(len(testfiles)):
            tokens = testfiles[i].split()
            vid_path = data_path + tokens[0] + "_uncomp.avi"
            while True:
                try:
                    vid = imageio.get_reader(vid_path, "ffmpeg")
                    break
                except Exception:
                    print("imageio failed loading frames, retrying")

            action = vid_path.split("_")[1]
            if action in ["running", "jogging"]:
                n_skip = 3
            else:
                n_skip = T
            start = int(tokens[1])
            end = int(tokens[2]) - length - 1
            if gif_per_vid != '-1':
                end = min(end, max(start + 1, start + (gif_per_vid - 1) * n_skip + 1))
            for j in xrange(start, end, n_skip):
                print("Video " + str(i) + "/" + str(len(testfiles)) + ". Index " + str(j) +
                      "/" + str(vid.get_length() - length - 1))

                folder_pref = vid_path.split("/")[-1].split(".")[0]
                folder_name = folder_pref + "." + str(j) + "-" + str(j + T)

                vid_names.append(folder_name)
                savedir = "../../results/images/KTH/" + prefix + "/" + folder_name

                seq_batch = np.zeros((1, image_size_h, image_size_w,
                                      length, c_dim), dtype="float32")
                for t in xrange(length):

                    # imageio fails randomly sometimes
                    while True:
                        try:
                            img = cv2.resize(vid.get_data(j + t), (image_size_h, image_size_w))
                            break
                        except Exception:
                            print("imageio failed loading frames, retrying")

                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    seq_batch[0, :, :, t] = transform(img[:, :, None])

                true_data = seq_batch.copy()
                pred_data = np.zeros(true_data.shape, dtype="float32")
                seq_batch_tran = create_missing_frames(seq_batch.transpose([0, 3, 1, 2, 4]), K, T)
                forward_seq = seq_batch_tran

                conv_layer_weight = np.zeros((convlstm_layer_num), dtype=np.float)
                conv_layer_weight[-1] = 1
                pred_data[0] = sess.run(model.G,
                                        feed_dict={model.forward_seq: forward_seq,
                                                   model.target: seq_batch,
                                                   model.conv_layer_weight: conv_layer_weight,
                                                   model.conv_layer_index: convlstm_layer_num - 1,
                                                   model.loss_reduce_weight: conv_layer_weight[convlstm_layer_num - 1],
                                                   model.is_dis: False,
                                                   model.is_gen: False})
                if not os.path.exists(savedir):
                    os.makedirs(savedir)

                cpsnr = np.zeros((length,))
                cssim = np.zeros((length,))
                for t in xrange(length):
                    pred = (inverse_transform(pred_data[0, :, :, t]) * 255).astype("uint8")
                    target = (inverse_transform(true_data[0, :, :, t]) * 255).astype("uint8")
                    if quantitative:
                        cpsnr[t] = measure.compare_psnr(pred, target)
                        cssim[t] = ssim.compute_ssim(Image.fromarray(cv2.cvtColor(target,
                                                                                  cv2.COLOR_GRAY2BGR)),
                                                     Image.fromarray(cv2.cvtColor(pred,
                                                                                  cv2.COLOR_GRAY2BGR)))
                    # pred = draw_frame(cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR), t % (T + K) < K)
                    if qualitative:
                        blank = (inverse_transform(seq_batch_tran[0, t, :, :]) * 255).astype("uint8")

                        cv2.imwrite(savedir + "/pred_" + "{0:04d}".format(t) + ".png", pred)
                        cv2.imwrite(savedir + "/gt_" + "{0:04d}".format(t) + ".png", target)
                        cv2.imwrite(savedir + "/blk_gt_" + "{0:04d}".format(t) + ".png", blank)
                if qualitative:
                    cmd1 = "rm " + savedir + "/pred.gif"
                    cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + savedir +
                            "/pred_%04d.png " + savedir + "/pred.gif")
                    cmd3 = "rm " + savedir + "/pred*.png"

                    # Comment out "system(cmd3)" if you want to keep the output images
                    # Otherwise only the gifs will be kept
                    system(cmd1);
                    system(cmd2); system(cmd3);

                    cmd1 = "rm " + savedir + "/gt.gif"
                    cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + savedir +
                            "/gt_%04d.png " + savedir + "/gt.gif")
                    cmd3 = "rm " + savedir + "/gt*.png"
                    # Comment out "system(cmd3)" if you want to keep the output images
                    # Otherwise only the gifs will be kept
                    system(cmd1);
                    system(cmd2);   system(cmd3);

                    cmd1 = "rm " + savedir + "/blk_gt.gif"
                    cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + savedir +
                            "/blk_gt_%04d.png " + savedir + "/blk_gt.gif")
                    cmd3 = "rm " + savedir + "/blk_gt*.png"

                    system(cmd1);
                    system(cmd2);  system(cmd3);
                if quantitative:
                    print psnr_err.shape, cpsnr.shape
                    print ssim_err.shape, cssim.shape
                    psnr_err = np.concatenate((psnr_err, cpsnr[None,:]), axis=0)
                    ssim_err = np.concatenate((ssim_err, cssim[None,:]), axis=0)

        if quantitative:
            np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
        print("Results saved to " + save_path)
        print "PSNR average:", np.mean(psnr_err), "SSIM average", np.mean(ssim_err)
    print("Done.")


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
    parser.add_argument("--pretrain_model", type=str, dest="pretrain_model",
                        default="", help="")
    parser.add_argument("--concatconv", action="store_true", dest="concatconv", help="concatconv")
    parser.add_argument("--gif_per_vid", type=int, dest="gif_per_vid", default=1,
                        help="refer to how many per video")
    parser.add_argument("--mode", type=str, dest="mode",
                        default="bi_one_serial", help="number frames to be sent in discriminator")
    parser.add_argument("--quantitative", action="store_true", dest="quantitative", help="quantitative")
    parser.add_argument("--qualitative", action="store_true", dest="qualitative", help="qualitative")
    args = parser.parse_args()

    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    # print_tensors_in_checkpoint_file(
    #     file_name='../../models/KTH/KTH_convlstm_image_size_h=128_image_size_w=128_K=1_T=3_B=5_convlstm_layer_num=2_dis_length=11_batch_size=16_alpha=1.0_beta=0.02_lr=0.0001_mode=bi_one_serial_fade_in=False_no_d=True_use_gt=True_res_mode=mix_pixel_loss=l2_concatconv=True_Unet=True/bi_conv_lstm.model-100001',
    #     tensor_name='', all_tensors=False)
    # exit()
    main(**vars(args))
