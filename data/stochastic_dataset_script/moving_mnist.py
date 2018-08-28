from __future__ import division
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

class MovingMNIST(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=True):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False  # multi threaded loading
        self.channels = 1
        self.train = train
        mnist = input_data.read_data_sets("../../../data/smmnist/", one_hot=False)
        if train:
            images = np.reshape(np.uint8(mnist.train.images * 255), (-1, 28, 28))
        else:
            images = np.reshape(np.uint8(mnist.test.images * 255), (-1, 28, 28))
            self.limit = 768
            print("test smnist")
        self.image_data = [np.array(Image.fromarray(images[i,:,:]).resize((self.digit_size, self.digit_size), Image.ANTIALIAS))
                           for i in range(images.shape[0])]
        print "np.max(images), np.min(images)", np.max(images), np.min(images)
        print "np.max(self.image_data), np.min(self.image_data)", np.max(self.image_data), np.min(self.image_data)
        print "len(self.image_data), self.image_data[0].shape", len(self.image_data), self.image_data[0].shape
        self.N = len(self.image_data)


    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if not self.train:
            if index >= self.limit:
                return None
            self.set_seed(index)
        else:
            self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((image_size,
                      image_size,
                      self.seq_len,
                      self.channels),
                     dtype=np.int32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit = self.image_data[idx]
            # print "digit.dtype, digit.shape", digit.dtype, digit.shape
            sx = np.random.randint(image_size - digit_size)
            sy = np.random.randint(image_size - digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size - 32:
                    sy = image_size - 32 - 1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size - 32:
                    sx = image_size - 32 - 1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)

                x[sy:sy + 32, sx:sx + 32, t, 0] += digit.squeeze()
                sy += dy
                sx += dx

        x[x > 255] = 255
        return x.astype(np.uint8)