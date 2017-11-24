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

def main(image_size, K, T, parent_dir, color_channel_num, gif_per_vid):
    data_path = "../../../data/KTH/"
    f = open(data_path + "test_data_list.txt", "r")
    testfiles = f.readlines()
    c_dim = 3
    prefix = ("gt")
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
      start = int(tokens[1])
      end = int(tokens[2])-2*K-T-1
      if gif_per_vid != '-1':
          end = min(end, max(start + 1, start + (gif_per_vid - 1) * n_skip + 1))
      for j in xrange(start,end,n_skip):
        print("Video "+str(i)+"/"+str(len(testfiles))+". Index "+str(j)+
              "/"+str(vid.get_length()-T-1))

        folder_pref = vid_path.split("/")[-1].split(".")[0]
        folder_name = folder_pref+"."+str(j)+"-"+str(j+T)

        savedir = args.parent_dir+prefix+"/"+folder_name
        if not exists(savedir):
            makedirs(savedir)
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
          img = transform(img[:,:,None])
          gt_frame = inverse_transform(img).astype("uint8")
          cv2.imwrite(savedir + "/gt_" + "{0:04d}".format(t) + ".png", gt_frame)

        cmd1 = "rm " + savedir + "/gt.gif"
        cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + savedir +
                "/gt_%04d.png " + savedir + "/gt.gif")
        system(cmd1)
        system(cmd2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=128, help="Pre-trained model")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of input images")
    parser.add_argument("--T", type=int, dest="T",
                        default=20, help="Number of steps into the future")
    parser.add_argument("--parent_dir", type=str, nargs="?", dest="parent_dir",
                        default="../../../tf_record/KTH/", help="parent_dir")
    parser.add_argument("--color_channel_num", type=int, dest="color_channel_num",
                        default=1, help="number of color channels")
    parser.add_argument("--gif_per_vid", type=int, dest="gif_per_vid", default=1,
                        help="refer to how many per video")
    args = parser.parse_args()
    main(**vars(args))