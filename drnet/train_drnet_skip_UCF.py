import sys
import time
import tensorflow as tf
import glob
sys.path.append("models")
from os import listdir, makedirs, system
from os.path import exists
from unet_128 import *
from argparse import ArgumentParser
import load_save_data
from dr_utils import *


parser = ArgumentParser()
parser.add_argument("--learningRate", type=float, dest="learningRate",
                    default=0.002, help="")
parser.add_argument("--beta1", type=float, dest="beta1",
                    default=0.02, help="")
parser.add_argument("--batch_size", type=int, dest="batch_size",
                    default=32, help="Mini-batch size")
parser.add_argument("--gpu", type=int, nargs="?", dest="gpu", default=0,
                    help="GPU device id")
parser.add_argument("--save", type=str, nargs="?", dest="save",
                    default="logs/", help="")
parser.add_argument("--name", type=str, nargs="?", dest="name",
                    default="default", help="")
parser.add_argument("--dataRoot", type=str, nargs="?", dest="dataRoot",
                    default="/path/to/data/", help="")
parser.add_argument("--optimizer", type=str, nargs="?", dest="optimizer",
                    default="adam", help="")
parser.add_argument("--nEpochs", type=int, dest="nEpochs",
                    default=300, help="")
parser.add_argument("--seed", type=int, dest="seed",
                    default=1, help="")
parser.add_argument("--epochSize", type=int, dest="epochSize",
                    default=50000, help="")
parser.add_argument("--plot_epoch", type=int, dest="plot_epoch",
                    default=100, help="")
parser.add_argument("--save_epoch", type=int, dest="save_epoch",
                    default=500, help="")
parser.add_argument("--contentDim", type=int, dest="contentDim",
                    default=128, help="")
parser.add_argument("--poseDim", type=int, dest="poseDim",
                    default=5, help="")
parser.add_argument("--image_size_h", type=int, dest="image_size_h",
                    default=128, help="")
parser.add_argument("--image_size_w", type=int, dest="image_size_w",
                    default=128, help="")
parser.add_argument("--dataset", type=str, nargs="?", dest="dataset",
                    default="kth", help="")
parser.add_argument("--maxStep", type=int, dest="maxStep",
                    default=12, help="")
parser.add_argument("--nShare", type=int, dest="nShare",
                    default=1, help="")
parser.add_argument("--advWeight", type=float, dest="advWeight",
                    default=0.0, help="")
parser.add_argument("--normalize", action="store_true", dest="normalize", help="")
parser.add_argument("--load_pretrain", action="store_true", dest="load_pretrain", help="")
parser.add_argument("--only_rec", action="store_true", dest="only_rec", help="")
parser.add_argument("--model", type=str, nargs="?", dest="model",
                    default="dcgan", help="")
# dcgan or vgg
parser.add_argument("--unet", type=str, nargs="?", dest="unet",
                    default="dcgan", help="")
# dcgan or resnet
parser.add_argument("--pose_net", type=str, nargs="?", dest="pose_net",
                    default="dcgan", help="")

parser.add_argument("--nThreads", type=int, dest="nThreads",
                    default=0, help="")
parser.add_argument("--dataPool", type=int, dest="dataPool",
                    default=200, help="")
parser.add_argument("--dataWarmup", type=int, dest="dataWarmup",
                    default=10, help="")
parser.add_argument("--beta", type=float, dest="beta",
                    default=0.0001, help="")
parser.add_argument("--color_channel_num", type=int, dest="color_channel_num",
                    default=1, help="")
parser.add_argument("--tf_record_train_dir", type=str, nargs="?", dest="tf_record_train_dir",
                        default="../tf_record/KTH/train/", help="tf_record train location")
parser.add_argument("--tf_record_test_dir", type=str, nargs="?", dest="tf_record_test_dir",
                        default="../tf_record/KTH/test/", help="tf_record test location")
args = parser.parse_args()

def get_first(vids, actions):
    gathered1 = []
    gathered2 = []
    i = 0
    while len(gathered1) < args.batch_size:
        for action in actions:
            gathered1.append(vids[action][i])
            gathered2.append(vids[action][-i - 1])
            if len(gathered1) >= args.batch_size: break
        i += 1
    # print "gathered1 + gathered2 len:", len(gathered1 + gathered2)
    return gathered1 + gathered2




iters = 0
prefix = ("KTH_drnet"
          + "_image_size_h=" + str(args.image_size_h)
          + "_image_size_w=" + str(args.image_size_w)
          + "_nShare=" + str(args.nShare)
          + "_batch_size=" + str(args.batch_size)
          + "_beta=" + str(args.beta)
          + "_normalize=" + str(args.normalize)
          + "_unet=" + str(args.unet)
          + "_pose_net=" + str(args.pose_net)
          + "_lr=" + str(args.learningRate))

print("\n" + prefix + "\n")
checkpoint_dir = "model_checkpoint/KTH/" + prefix + "/"
samples_dir = "samples/KTH/" + prefix + "/"
summary_dir = "logs/KTH/" + prefix + "/"

if not exists(checkpoint_dir):
    makedirs(checkpoint_dir)
#     save synthesized frame sample
if not exists(samples_dir):
    makedirs(samples_dir)
if not exists(summary_dir):
    makedirs(summary_dir)

device_string = "/gpu:%d" % args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
with tf.device(device_string):
    model = UnetDCGAN(args)
    model.build_model()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gen_loss = model.gen_loss
        if args.only_rec: gen_loss = model.rec_loss
        optimizer_gen = tf.train.AdamOptimizer(args.learningRate, beta1=args.beta1).minimize(
            gen_loss, var_list=model.g_vars
        )
        optimizer_dis = tf.train.AdamOptimizer(args.learningRate, beta1=args.beta1).minimize(
            model.entropy_loss, var_list=model.d_vars
        )
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False)
config.gpu_options.allow_growth = False

train_vids = load_save_data.load_records_with_action(
    glob.glob(args.tf_record_train_dir + '*tfrecords*'))
test_vids = load_save_data.load_records_with_action(
    glob.glob(args.tf_record_test_dir + '*tfrecords*'))
train_action_keys = train_vids.keys()
test_action_keys = test_vids.keys()

plot_train_seq_batch, plot_train_seq_batch_same, plot_train_seq_batch_diff = load_triplet_data_from_list(
    get_first(train_vids,train_action_keys), range(args.batch_size),range(args.batch_size, args.batch_size * 2),
    args.image_size_h, args.image_size_w, args.nShare, args.maxStep, flipable=False, channel=args.color_channel_num)
plot_test_seq_batch, plot_test_seq_batch_same, plot_test_seq_batch_diff = load_triplet_data_from_list(
    get_first(test_vids,train_action_keys), range(args.batch_size),range(args.batch_size, args.batch_size * 2),
    args.image_size_h, args.image_size_w, args.nShare, args.maxStep, flipable=False, channel=args.color_channel_num)
same_array = np.ones((args.batch_size,1))
diff_array = np.zeros((args.batch_size,1))
half_array = (same_array - diff_array ) / 2.0
# print same_array,diff_array,half_array
with tf.Session(config=config) as sess:

    tf.global_variables_initializer().run()
    if args.load_pretrain:
        if model.load(sess, checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    writer = tf.summary.FileWriter(summary_dir, sess.graph)

    epoch = 0
    start_time = time.time()

    while epoch < args.epochSize:
        random.shuffle(train_action_keys)
        for action in train_action_keys:
            mini_batches = get_minibatches_idx(len(train_vids[action]), args.batch_size, shuffle=True)
            print "epoch:", epoch, ",start trained action:", action
            for _, batchidx, alter_batchidx in mini_batches:
                seq_batch, seq_batch_same, seq_batch_diff = load_triplet_data_from_list(
                    train_vids[action], batchidx, alter_batchidx, args.image_size_h, args.image_size_w,
                    args.nShare, args.maxStep, flipable=True, channel = args.color_channel_num)
                # update D
                # print same_array[:seq_batch.shape[0],:].shape
                _, dis_loss, summary_str = sess.run([optimizer_dis, model.entropy_loss, model.entropy_loss_min_sum],
                                          feed_dict={model.x_same_1: seq_batch,
                                                     model.x_same_2: seq_batch_same,
                                                     model.x_diff_3: seq_batch_diff,
                                                     model.same_label: same_array[:seq_batch.shape[0],:],
                                                     model.diff_label: diff_array[:seq_batch.shape[0],:],
                                                     model.is_dis: True,
                                                     model.is_gen: True})
                writer.add_summary(summary_str, iters + 1)
                # update G
                _, hc_same_loss,entropy_loss_max,rec_loss, summary_str \
                    = sess.run([optimizer_gen, model.hc_same_loss, model.entropy_loss,
                                model.rec_loss, model.gen_merge_sum],
                                                    feed_dict={model.x_same_1: seq_batch,
                                                               model.x_same_2: seq_batch_same,
                                                               model.x_diff_3: seq_batch_diff,
                                                               model.same_label: half_array[:seq_batch.shape[0],:],
                                                               model.diff_label: half_array[:seq_batch.shape[0],:],
                                                               model.is_dis: True,
                                                               model.is_gen: True
                                                               })
                writer.add_summary(summary_str, iters + 1)
                print("Iters: [%2d] time: %4.4f, hc_same_loss: %.8f, dis_loss: %.8f, entropy_loss_max: %.8f, rec_loss: %.8f"
                        % (iters, time.time() - start_time, hc_same_loss, dis_loss, entropy_loss_max, rec_loss))
                iters += 1
            print "epoch:",epoch, ",finish trained action:", action
        epoch += 1
        if np.mod(epoch, args.plot_epoch) == 1:
            plot_train_pred3_1, plot_train_pred1, plot_train_1, plot_train_2, plot_train_3, rec_loss_train \
                = sess.run([model.pred3_1, model.pred1, model.x_single_1, model.x_single_2, model.x_single_3,
                            model.rec_loss,],
                            feed_dict={model.x_same_1: plot_train_seq_batch,
                                       model.x_same_2: plot_train_seq_batch_same,
                                       model.x_diff_3: plot_train_seq_batch_diff,
                                       model.is_dis: False,
                                       model.is_gen: False})

            plot_test_pred3_1, plot_test_pred1, plot_test_1, plot_test_2, plot_test_3, rec_loss_test\
                = sess.run([model.pred3_1, model.pred1, model.x_single_1, model.x_single_2,model.x_single_3,
                            model.rec_loss,],
                            feed_dict={model.x_same_1: plot_train_seq_batch,
                                       model.x_same_2: plot_train_seq_batch_same,
                                       model.x_diff_3: plot_train_seq_batch_diff,
                                       model.is_dis: False,
                                       model.is_gen: False})

            for i in range(args.batch_size // 5):
                if i == 0:
                    sample_train_cat = np.concatenate((plot_train_1[i], plot_train_2[i], plot_train_pred1[i], plot_train_3[i], plot_train_pred3_1[i]), axis=0)
                    sample_test_cat = np.concatenate((plot_test_1[i], plot_test_2[i], plot_test_pred1[i], plot_test_3[i], plot_test_pred3_1[i]), axis=0)
                else:
                    sample_train_cat = np.concatenate((sample_train_cat,
                       np.concatenate((plot_train_1[i], plot_train_2[i], plot_train_pred1[i], plot_train_3[i], plot_train_pred3_1[i]), axis=0)), axis=1)
                    sample_test_cat = np.concatenate((sample_test_cat,
                       np.concatenate((plot_test_1[i], plot_test_2[i], plot_test_pred1[i], plot_test_3[i], plot_test_pred3_1[i]), axis=0)), axis=1)
            print(
                    "plot trainset, Iters: [%2d] time: %4.4f, rec_loss: %.8f"
                    % (iters, time.time() - start_time, rec_loss_train)
            )

            print(
                    "plot testset, Iters: [%2d] time: %4.4f, rec_loss_sum: %.8f"
                    % (iters, time.time() - start_time, rec_loss_test)
            )
            summary = sess.run(model.summary_merge_plot_img, feed_dict={
                model.train_seq_img: np.expand_dims(inverse_transform(sample_train_cat) * 255, axis=0),
                model.test_seq_img: np.expand_dims(inverse_transform(sample_test_cat) * 255, axis=0)
            })
            writer.add_summary(summary, iters)

        if np.mod(epoch, args.save_epoch) == 0:
            model.save(sess, checkpoint_dir, iters)






