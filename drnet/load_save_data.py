import tensorflow as tf
import numpy as np
import imageio
import cv2
import dr_utils
import glob

def vids_2_frames(tfiles, data_path, image_size_h, image_size_w, channel):
    vids=[]
    actions=[]
    for i in xrange(len(tfiles)):
        f_name = tfiles[i]
        tokens = f_name.split()
        vid_path = data_path + tokens[0] + ("_uncomp.avi" if channel ==1 else "")
        print "vid_path:", vid_path
        try:
            vid = imageio.get_reader(vid_path, "ffmpeg")
            if len(tokens) < 2:
                low = 1
                high = vid.get_length()
            else:
                low = int(tokens[1])
                high = np.min([int(tokens[2]), vid.get_length()])
            seq = np.zeros((image_size_h, image_size_w, high - low + 1, channel), dtype=np.uint8)
            for t in xrange(high - low + 1):
                if channel == 1:
                    # w,h not h,w here!
                    img = cv2.cvtColor(cv2.resize(vid.get_data(t),
                                                  (image_size_w, image_size_h)), cv2.COLOR_RGB2GRAY)
                else:
                    img = cv2.resize(vid.get_data(t),(image_size_w, image_size_h))
                # print img.shape, seq.shape
                seq[:, :, t, :] = img[:, :, :]
            if (len(tokens[0].split("_"))) > 0:
                actions.append(tokens[0].split("_")[1])
            vids.append(seq)
        except:
            continue

    return vids, actions

def load_records_with_action(tf_record_files):
    vids = {}
    count = 0
    for i in xrange(len(tf_record_files)):
        print "loading {}".format(tf_record_files[i])
        record_iterator = tf.python_io.tf_record_iterator(path=tf_record_files[i])
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            height = int(example.features.feature['height']
                         .int64_list
                         .value[0])

            width = int(example.features.feature['width']
                        .int64_list
                        .value[0])

            depth = int(example.features.feature['depth']
                        .int64_list
                        .value[0])

            vid_string = (example.features.feature['vid']
                          .bytes_list
                          .value[0])

            action_string = (example.features.feature['action']
                          .bytes_list
                          .value[0])

            vid_raw = np.fromstring(vid_string, dtype=np.uint8)
            action = action_string.decode('UTF-8')
            vid = vid_raw.reshape((height, width, depth, -1))
            if action in vids:
                vids[action].append(vid)
            else:
                vids[action] = [vid]
            count += 1
        print "finish {}".format(tf_record_files[i])
    print count, " videos in total"
    return vids

def load_record_inbatch(file_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(serialized_example,
        features={
            'depth': tf.FixedLenFeature([1], tf.int64),
            'vid': tf.FixedLenFeature([],tf.string),
        }
    )
    depth = tf.cast(features['depth'], tf.int32)
    return features["vid"], depth

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
