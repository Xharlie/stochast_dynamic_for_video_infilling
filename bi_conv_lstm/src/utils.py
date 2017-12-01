"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import cv2
import random
import imageio
import scipy.misc
import numpy as np
import os

def transform(image):
    return image/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


def save_images(images, size, image_path):
  return imsave(inverse_transform(images)*255., size, image_path)


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx / size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img


def imsave(images, size, path):
  return scipy.misc.imsave(path, merge(images, size))


def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """ 
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0 
  for i in range(n // minibatch_size):
    minibatches.append(idx_list[minibatch_start:
                                minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n): 
    # Make a minibatch out of what is left
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)


def draw_frame(img, is_input):
  if img.shape[2] == 1:
    img = np.repeat(img, [3], axis=2)

  if is_input:
    img[:2,:,0]  = img[:2,:,2] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,2] = 0 
    img[:,-2:,0] = img[:,-2:,2] = 0 
    img[:2,:,1]  = 255 
    img[:,:2,1]  = 255 
    img[-2:,:,1] = 255 
    img[:,-2:,1] = 255 
  else:
    img[:2,:,0]  = img[:2,:,1] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,1] = 0 
    img[:,-2:,0] = img[:,-2:,1] = 0 
    img[:2,:,2]  = 255 
    img[:,:2,2]  = 255 
    img[-2:,:,2] = 255 
    img[:,-2:,2] = 255 

  return img 

def draw_blank(img, is_input):
  if not is_input:
    img = np.zeros_like(img)
  return img

# get seq and diff start arbitrarially
def load_kth_data(f_name, data_path, image_size, K, T): 
  flip = np.random.binomial(1,.5,1)[0]
  tokens = f_name.split()
  vid_path = data_path + tokens[0] + "_uncomp.avi"
  vid = imageio.get_reader(vid_path,"ffmpeg")
  low = int(tokens[1])
  high = np.min([int(tokens[2]),vid.get_length()])- 2 * K - T + 1
  if low == high:
    stidx = 0 
  else:
    if low >= high: print(vid_path)
    stidx = np.random.randint(low=low, high=high)
  seq = np.zeros((image_size, image_size, 2*K+T, 1), dtype="float32")
  for t in xrange(2*K+T):
    img = cv2.cvtColor(cv2.resize(vid.get_data(stidx+t),
                       (image_size,image_size)),
                       cv2.COLOR_RGB2GRAY)
    seq[:,:,t] = transform(img[:,:,None])

  if flip == 1:
    seq = seq[:,::-1]

  return seq

def load_kth_data_from_list(train_vids, batchidx, image_size, K, T, B, flipable=True):
  length = B * (K + T) + K
  seq = np.zeros((len(batchidx), image_size, image_size, length, 1), dtype="float32")
  for i in range(len(batchidx)):
    stidx = 0
    selected = train_vids[batchidx[i]]
    while True:
      low = 0
      high = selected.shape[2]- length
      if low == high:
        stidx = 0
      elif high < 0:
        selected = train_vids[np.random.randint(low=0, high=len(train_vids))]
        continue
      else:
        stidx = np.random.randint(low=low, high=high)
      break
    for t in xrange(length):
      seq[i,:,:,t] = transform(selected[:,:,t+stidx,:])

    if flipable:
      flip = np.random.binomial(1,.5,1)[0]
    else:
      flip = 0
    if flip == 1:
      seq = seq[:,::-1]
  return seq

def load_s1m_data(f_name, data_path, trainlist, K, T):
  flip = np.random.binomial(1,.5,1)[0]
  vid_path = data_path + f_name
  img_size = [240,320]

  while True:
    try:
      vid = imageio.get_reader(vid_path,"ffmpeg")
      low = 1
      high = vid.get_length()-K-T+1
      if low == high:
        stidx = 0
      else:
        stidx = np.random.randint(low=low, high=high)
      seq = np.zeros((img_size[0], img_size[1], K+T, 3),
                     dtype="float32")
      for t in xrange(K+T):
        img = cv2.resize(vid.get_data(stidx+t),
                         (img_size[1],img_size[0]))[:,:,::-1]
        seq[:,:,t] = transform(img)

      if flip == 1:
        seq = seq[:,::-1]

      diff = np.zeros((img_size[0], img_size[1], K-1, 1),
                      dtype="float32")
      for t in xrange(1,K):
        prev = inverse_transform(seq[:,:,t-1])*255
        prev = cv2.cvtColor(prev.astype("uint8"),cv2.COLOR_BGR2GRAY)
        next = inverse_transform(seq[:,:,t])*255
        next = cv2.cvtColor(next.astype("uint8"),cv2.COLOR_BGR2GRAY)
        diff[:,:,t-1,0] = (next.astype("float32")-prev.astype("float32"))/255.
      break
    except Exception:
      # In case the current video is bad load a random one 
      rep_idx = np.random.randint(low=0, high=len(trainlist))
      f_name = trainlist[rep_idx]
      vid_path = data_path + f_name

  return seq, diff


def check_create_dir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)
    return dir

def create_missing_frames(seq_batch_tran, K,T):
    for i in range(seq_batch_tran.shape[1]):
      modulus = np.mod(i, K+T)
      if modulus >= K:
        seq_batch_tran[:,i,:,:,:] = np.zeros_like(seq_batch_tran[:,i,:,:,:])
    return seq_batch_tran