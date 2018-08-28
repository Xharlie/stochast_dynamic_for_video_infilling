import random
import os
import numpy as np
from scipy import misc
from torch.utils.serialization import load_lua


class KTH_STO(object):

    def __init__(self, train, data_root, seq_len=20, image_size=64):
        self.data_root = '%s/processed' % data_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

        self.dirs = os.listdir(self.data_root)
        if train:
            self.train = True
            data_type = 'train'
            self.persons = list(range(1, 21))
        else:
            self.train = False
            self.persons = list(range(21, 26))
            data_type = 'test'

        self.data = {}
        for c in self.classes:
            self.data[c] = load_lua('%s/%s/%s_meta%dx%d.t7' % (self.data_root, c, data_type, image_size, image_size))
            print(c,"'s length:",len(self.data[c]))
        self.seed_set = False
        self.sample_per_vid = 2
        self.current_sample_indx = 0
        self.current_class_index = 0
        self.current_vid_index = [0, 0, 0, 0, 0, 0]

    def get_sequence(self):
        t = self.seq_len
        if self.train:
            while True:  # skip seqeunces that are too short
                c_idx = np.random.randint(len(self.classes))
                c = self.classes[c_idx]
                vid_idx = np.random.randint(len(self.data[c]))
                vid = self.data[c][vid_idx]
                seq_idx = np.random.randint(len(vid['files']))
                if len(vid['files'][seq_idx]) - t >= 0:
                    break
        else:
            while True:  # skip seqeunces that are too short
                c = self.classes[self.current_class_index]
                if self.current_vid_index[self.current_class_index] >= len(self.data[c]):
                    self.current_class_index += 1
                    print("finish class", self.current_vid_index[self.current_class_index-1])
                    if self.current_class_index >= len(self.classes):
                        return None
                    c = self.classes[self.current_class_index]
                vid_idx = self.current_vid_index[self.current_class_index]
                vid = self.data[c][vid_idx]
                seq_idx = np.random.randint(len(vid['files']))
                if len(vid['files'][seq_idx]) - t >= 0:
                    self.current_sample_indx += 1
                    if self.current_sample_indx >= self.sample_per_vid:
                        self.current_sample_indx = 0
                        self.current_vid_index[self.current_class_index] += 1
                    break
        dname = '%s/%s/%s' % (self.data_root, c, vid['vid'])
        st = random.randint(0, len(vid['files'][seq_idx]) - t)

        seq = []
        for i in range(st, st + t):
            fname = '%s/%s' % (dname, vid['files'][seq_idx][i])
            im = misc.imread(fname)
            im = np.expand_dims(np.array(im[:, :, 0].reshape(self.image_size, self.image_size, 1)), axis=2)
            seq.append(im)
        return np.concatenate(seq, axis=2)

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            # torch.manual_seed(index)
        return self.get_sequence()

    def __len__(self):
        return len(self.dirs) * 36 * 5  # arbitrary