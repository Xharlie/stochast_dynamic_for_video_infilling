import tensorflow as tf
import numpy as np
import imageio
import cv2

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
                if len(img.shape) == 2: seq[:, :, t, :] = img[:, :, None]
                else: seq[:, :, t, :] = img[:, :, :]
            print tokens[0]
            if (len(tokens[0].split("_"))) > 0:
                actions.append(tokens[0].split("_")[1])
            vids.append(seq)
        except KeyError:
            print KeyError.message
            continue

    return vids, actions


def save_data2record(tfiles, data_path, image_size_h, image_size_w, tf_record_dir, channel):
    vids, actions = vids_2_frames(tfiles, data_path, image_size_h, image_size_w, channel)
    print actions
    tfrecords_filename = 'tfrecords'
    writer = tf.python_io.TFRecordWriter(tf_record_dir + tfrecords_filename)
    for i in xrange(len(vids)):
        vids_record = tf.train.Example(features=tf.train.Features(
            feature={
                'height': _int64_feature(image_size_h),
                'width': _int64_feature(image_size_w),
                'depth': _int64_feature(vids[i].shape[2]),
                'channels': _int64_feature(channel),
                'action': _bytes_feature(actions[i]),
                'vid': _bytes_feature(vids[i].tostring())
            }
        ))
        writer.write(vids_record.SerializeToString())
        print "finish writing video{} to {}".format(i, tf_record_dir + tfrecords_filename)
    writer.close()
    return vids

def save_data2records(tfiles, data_path, image_size_h, image_size_w, tf_record_dir, channel):
    tf_size = 800
    start = 0
    end = 0
    files=[]
    while start <= len(tfiles):
        end = min(start + tf_size, len(tfiles) + 1)
        if end + tf_size / 4 > len(tfiles): end = len(tfiles) + 1
        print "file start and end:",start,end
        vids, actions = vids_2_frames(tfiles[start:end], data_path, image_size_h, image_size_w, channel)
        tfrecords_filename = 'tfrecords' + str(start / tf_size)
        writer = tf.python_io.TFRecordWriter(tf_record_dir + tfrecords_filename)
        for i in xrange(len(vids)):
            vids_record = tf.train.Example(features=tf.train.Features(
                feature={
                    'height': _int64_feature(image_size_h),
                    'width': _int64_feature(image_size_w),
                    'depth': _int64_feature(vids[i].shape[2]),
                    'channels': _int64_feature(channel),
                    'action': _bytes_feature(actions[i]),
                    'vid': _bytes_feature(vids[i].tostring())
                }
            ))
            writer.write(vids_record.SerializeToString())
            print "finish writing video{} to {}".format(i, tf_record_dir + tfrecords_filename)
            files.append(tf_record_dir + tfrecords_filename)
        writer.close()
        start = end
    return files

def load_records(tf_record_files, length=None):
    vids = []
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
            if length is not None and vid.shape[-2] < length:
                print length, vid.shape[-2]
                continue
            vids.append(vid)
        print "finish {}".format(tf_record_files[i])
        print len(vids), " videos in total"
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
