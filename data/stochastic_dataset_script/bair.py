import os
import io
from scipy.misc import imresize
import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread
from joblib import Parallel, delayed

class RobotPush(object):
    """Data Handler that loads robot pushing data."""

    def __init__(self, data_root, train=True, seq_len=20, image_size=64):
        self.root_dir = data_root
        if train:
            self.data_dir = '%s/processed_data/train' % self.root_dir
            self.ordered = False
        else:
            self.data_dir = '%s/processed_data/test' % self.root_dir
            self.ordered = True
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = seq_len
        self.image_size = image_size
        self.seed_is_set = False  # multi threaded loading
        self.d = 0

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return 10000

    def get_seq(self, index, parallel = False):
        if self.ordered:
            if index >= len(self.dirs):
                return None
            d = self.dirs[index]
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]
        image_seq = []
        fnames = []
        for i in range(self.seq_len):
            fnames.append('%s/%d.png' % (d, i))
        if parallel:
            with Parallel(n_jobs=12 / 3) as parallel:
                image_seq = parallel(delayed(self.read_frame)(fname)
                                  for fname in fnames)
        else:
            for fname in fnames:
                image_seq.append(self.read_frame(fname))
        return np.concatenate(image_seq, axis=2)

    def read_frame(self, fname):
        return np.asarray(imread(fname).reshape(64, 64, 1, 3))

    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq(index)