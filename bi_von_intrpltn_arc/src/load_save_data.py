import tensorflow as tf
import numpy as np
import imageio
import cv2
import utils
import glob

def vids_2_frames(tfiles, data_path, image_size,channel):
    vids=[]

    for i in xrange(len(tfiles)):
        f_name = tfiles[i]
        tokens = f_name.split()
        vid_path = data_path + tokens[0] + "_uncomp.avi"
        vid = imageio.get_reader(vid_path, "ffmpeg")
        low = int(tokens[1])
        high = np.min([int(tokens[2]), vid.get_length()])
        seq = np.zeros((image_size, image_size, high - low + 1, channel), dtype=np.uint8)
        for t in xrange(high - low + 1):
            img = cv2.cvtColor(cv2.resize(vid.get_data(t),
                                          (image_size, image_size)),
                               cv2.COLOR_RGB2GRAY)

            seq[:, :, t, :] = img[:, :, None]
        vids.append(seq)
    return vids


def save_kth_data2record(tfiles, data_path, image_size, tf_record_dir, channel):
    vids = vids_2_frames(tfiles, data_path, image_size,channel)
    tfrecords_filename = 'kth.tfrecords'
    writer = tf.python_io.TFRecordWriter(tf_record_dir + tfrecords_filename)
    for i in xrange(len(vids)):
        vids_record = tf.train.Example(features=tf.train.Features(
            feature={
                'height': _int64_feature(image_size),
                'width': _int64_feature(image_size),
                'depth': _int64_feature(vids[i].shape[2]),
                'channels': _int64_feature(channel),
                'vid': _bytes_feature(vids[i].tostring())
            }
        ))
        writer.write(vids_record.SerializeToString())
        print "finish writing video{} to {}".format(i, tf_record_dir + tfrecords_filename)
    writer.close()
    return vids

def load_kth_records(tf_record_dir):
    vids = []
    tf_record_files = glob.glob(tf_record_dir + '*.tfrecords')
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

            vid_raw = np.fromstring(vid_string, dtype=np.uint8)
            vid = vid_raw.reshape((height, width, depth, -1))
            vids.append(vid)
        print "finish {}".format(tf_record_files[i])
    return vids

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
