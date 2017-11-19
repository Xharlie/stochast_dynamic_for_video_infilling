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
from bi_von_model import bi_von_net
from utils import *


def main(lr, batch_size, alpha, beta, image_size, K, T, gpu,cpu, tf_record_test_dir, color_channel_num,
         fea_enc_model, dyn_enc_model, reference_mode):
  data_path = "../../../data/KTH/"
  f = open(data_path + "test_data_list.txt", "r")
  testfiles = f.readlines()
  c_dim = 1
  iters = 0

  best_model = None # will pick last model
  prefix = ("KTH_lstr"
            + "_image_size=" + str(image_size)
            + "_K=" + str(K)
            + "_T=" + str(T)
            + "_fea_enc_model=" + str(fea_enc_model)
            + "_dyn_enc_model=" + str(dyn_enc_model)
            + "_reference_mode=" + str(reference_mode)
            + "_batch_size=" + str(batch_size)
            + "_alpha=" + str(alpha)
            + "_beta=" + str(beta)
            + "_lr=" + str(lr))
  checkpoint_dir = "../../models/" + prefix + "/"
  device_string = ""
  if cpu:
    device_string = "/cpu:0"
  elif gpu:
    device_string = "/gpu:%d" % gpu[0]
  with tf.device(device_string):
    model = bi_von_net(image_size=[image_size,image_size], c_dim=1,
                  K=K, batch_size=1, T=T,
                  checkpoint_dir=checkpoint_dir, fea_enc_model=fea_enc_model,
                 dyn_enc_model=dyn_enc_model,
                 debug = False, reference_mode = reference_mode)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=gpu_options)) as sess:

    tf.global_variables_initializer().run()

    loaded, model_name = model.load(sess, checkpoint_dir, best_model)

    if loaded:
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed... exitting")
      return

    quant_dir = "../../results/quantitative/KTH/"+prefix+"/"
    save_path = quant_dir+"results_model="+model_name+".npz"
    if not exists(quant_dir):
      makedirs(quant_dir)


    vid_names = []
    psnr_err = np.zeros((0, T))
    ssim_err = np.zeros((0, T))
    for i in xrange(len(testfiles)):
      tokens = testfiles[i].split()
      vid_path = data_path+tokens[0]+"_uncomp.avi"
      while True:
        try:
          vid = imageio.get_reader(vid_path,"ffmpeg")
          break
        except Exception:
          print("imageio failed loading frames, retrying")

      action = vid_path.split("_")[1]
      if action in ["running", "jogging"]:
        n_skip = 3
      else:
        n_skip = T

      for j in xrange(int(tokens[1]),int(tokens[2])-2*K-T-1,n_skip):
        print("Video "+str(i)+"/"+str(len(testfiles))+". Index "+str(j)+
              "/"+str(vid.get_length()-T-1))

        folder_pref = vid_path.split("/")[-1].split(".")[0]
        folder_name = folder_pref+"."+str(j)+"-"+str(j+T)

        vid_names.append(folder_name)
        savedir = "../../results/images/KTH/"+prefix+"/"+folder_name

        seq_batch = np.zeros((1, image_size, image_size,
                              2 * K+T, c_dim), dtype="float32")
        for t in xrange(2*K+T):

          # imageio fails randomly sometimes
          while True:
            try:
              img = cv2.resize(vid.get_data(j+t), (image_size, image_size))
              break
            except Exception:
              print("imageio failed loading frames, retrying")

          img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
          seq_batch[0,:,:,t] = transform(img[:,:,None])

        true_data = seq_batch[:, :, :, K:K + T, :].copy()
        pred_data = np.zeros(true_data.shape, dtype="float32")
        seq_batch_tran = seq_batch.transpose([0, 3, 1, 2, 4])
        forward_seq = seq_batch_tran[:, :K, :, :, :]
        backward_seq = seq_batch_tran[:, ::-1][:, :K, :, :, :]

        xt = seq_batch[:,:,:,K-1]
        pred_data[0] = sess.run(model.G,
                                feed_dict={model.forward_seq: forward_seq,
                                          model.backward_seq: backward_seq,
                                          model.target: seq_batch})

        if not os.path.exists(savedir):
          os.makedirs(savedir)

        cpsnr = np.zeros((2*K+T,))
        cssim = np.zeros((2*K+T,))
        pred_data = np.concatenate((seq_batch[:,:,:,:K], pred_data, seq_batch[:,:,:,K+T:]),axis=3)
        true_data = np.concatenate((seq_batch[:,:,:,:K], true_data, seq_batch[:,:,:,K+T:]),axis=3)
        for t in xrange(2*K+T):
          pred = (inverse_transform(pred_data[0,:,:,t])*255).astype("uint8")
          target = (inverse_transform(true_data[0,:,:,t])*255).astype("uint8")

          cpsnr[t] = measure.compare_psnr(pred,target)
          cssim[t] = ssim.compute_ssim(Image.fromarray(cv2.cvtColor(target,
                                                       cv2.COLOR_GRAY2BGR)),
                                       Image.fromarray(cv2.cvtColor(pred,
                                                       cv2.COLOR_GRAY2BGR)))
          pred = draw_frame(pred, t < K or t >= K+T)
          target = draw_frame(target, t < K or t >= K+T)

          cv2.imwrite(savedir+"/pred_"+"{0:04d}".format(t)+".png", pred)
          cv2.imwrite(savedir+"/gt_"+"{0:04d}".format(t)+".png", target)

        cmd1 = "rm "+savedir+"/pred.gif"
        cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                "/pred_%04d.png "+savedir+"/pred.gif")
        cmd3 = "rm "+savedir+"/pred*.png"

        # Comment out "system(cmd3)" if you want to keep the output images
        # Otherwise only the gifs will be kept
        system(cmd1); system(cmd2); #system(cmd3);

        cmd1 = "rm "+savedir+"/gt.gif"
        cmd2 = ("ffmpeg -f image2 -framerate 7 -i "+savedir+
                "/gt_%04d.png "+savedir+"/gt.gif")
        cmd3 = "rm "+savedir+"/gt*.png"

        # Comment out "system(cmd3)" if you want to keep the output images
        # Otherwise only the gifs will be kept
        system(cmd1); system(cmd2); #system(cmd3);

        psnr_err = np.concatenate((psnr_err, cpsnr[None,K:]), axis=0)
        ssim_err = np.concatenate((ssim_err, cssim[None,K:]), axis=0)

    np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
    print("Results saved to "+save_path)
  print("Done.")

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.0001, help="Base Learning Rate")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=1, help="Mini-batch size")
  parser.add_argument("--alpha", type=float, dest="alpha",
                      default=1.0, help="Image loss weight")
  parser.add_argument("--beta", type=float, dest="beta",
                      default=0.02, help="GAN loss weight")
  parser.add_argument("--image_size", type=int, dest="image_size",
                      default=128, help="Pre-trained model")
  parser.add_argument("--K", type=int, dest="K",
                      default=10, help="Number of input images")
  parser.add_argument("--T", type=int, dest="T",
                      default=20, help="Number of steps into the future")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                      help="GPU device id")
  parser.add_argument("--cpu", action="store_true", dest="cpu", help="use cpu only")

  parser.add_argument("--tf_record_test_dir", type=str, nargs="?", dest="tf_record_test_dir",
                      default="../../../tf_record/KTH/test/", help="tf_record test location")
  parser.add_argument("--color_channel_num", type=int, dest="color_channel_num",
                      default=1, help="number of color channels")
  parser.add_argument("--fea_enc_model", type=str, dest="fea_enc_model", default="pooling",
                      help="feature extraction model")
  parser.add_argument("--dyn_enc_model", type=str, dest="dyn_enc_model", default="mix",
                      help="dynamic encoding model")
  parser.add_argument("--reference_mode", type=str, dest="reference_mode", default="two",
                      help="refer to how many frames in the end")
  args = parser.parse_args()
  main(**vars(args))
