import sys
import time
import tensorflow as tf
import glob

from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
sys.path.append("../")
import load_save_data
from bi_conv_lstm_model import bi_convlstm_net
from utils import *

def main(lr, batch_size, alpha, beta, image_size, K, T, B, convlstm_layer_num, num_iter, gpu, cpu,load_pretrain,
         tf_record_train_dir, tf_record_test_dir, color_channel_num, fea_enc_model,
                 dyn_enc_model, reference_mode, debug):
  check_create_dir(tf_record_train_dir)
  check_create_dir(tf_record_test_dir)
  train_tf_record_files=glob.glob(tf_record_train_dir+'*.tfrecords')
  test_tf_record_files=glob.glob(tf_record_test_dir+'*.tfrecords')

  train_vids = []
  if len(train_tf_record_files) == 0:
      data_path = "../../../data/KTH/"
      f = open(data_path+"train_data_list_trimmed.txt","r")
      trainfiles = f.readlines()
      train_vids = load_save_data.save_kth_data2record(
          trainfiles, data_path, image_size, tf_record_train_dir, color_channel_num)
  else:
      train_vids = load_save_data.load_kth_records(tf_record_train_dir)

  test_vids = []
  if len(test_tf_record_files) == 0:
      data_path = "../../../data/KTH/"
      f = open(data_path + "test_data_list.txt", "r")
      testfiles = f.readlines()
      test_vids = load_save_data.save_kth_data2record(
          testfiles, data_path, image_size, tf_record_test_dir, color_channel_num)
  else:
      test_vids = load_save_data.load_kth_records(tf_record_test_dir)

  margin = 0.3
  updateD = True
  updateG = True
  iters = 0
  prefix  = ("KTH_convlstm"
          + "_image_size="+str(image_size)
          + "_K="+str(K)
          + "_T="+str(T)
          + "_B="+str(B)
          + "_convlstm_layer_num="+str(convlstm_layer_num)
          + "_fea_enc_model="+str(fea_enc_model)
          + "_dyn_enc_model="+str(dyn_enc_model)
          + "_reference_mode="+str(reference_mode)
          + "_batch_size="+str(batch_size)
          + "_alpha="+str(alpha)
          + "_beta="+str(beta)
          + "_lr="+str(lr))

  print("\n"+prefix+"\n")
  checkpoint_dir = "../../models/"+prefix+"/"
  samples_dir = "../../samples/"+prefix+"/"
  summary_dir = "../../logs/"+prefix+"/"

  if not exists(checkpoint_dir):
    makedirs(checkpoint_dir)
  #     save synthesized frame sample
  if not exists(samples_dir):
    makedirs(samples_dir)
  if not exists(summary_dir):
    makedirs(summary_dir)

  device_string = ""
  if cpu:
      device_string = "/cpu:0"
  elif gpu:
      device_string = "/gpu:%d" % gpu[0]
  with tf.device(device_string):
    model = bi_convlstm_net(image_size=[image_size,image_size], c_dim=1,
                  K=K, T=T, B=B, convlstm_layer_num=convlstm_layer_num, batch_size=batch_size,
                  checkpoint_dir=checkpoint_dir, debug = debug, reference_mode = reference_mode)
    # for var in tf.trainable_variables():
    #     print var.name
    # raw_input('print trainable variables...')
    d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        model.d_loss, var_list=model.d_vars
    )
    g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        alpha * model.L_img + beta * model.L_GAN, var_list=model.g_vars
    )

  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
  config=tf.ConfigProto(allow_soft_placement=True,
                  log_device_placement=False)
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:

    tf.global_variables_initializer().run()

    if load_pretrain:
        if model.load(sess, checkpoint_dir):
          print(" [*] Load SUCCESS")
        else:
          print(" [!] Load failed...")

    g_sum = tf.summary.merge([model.L_p_sum,
                              model.L_gdl_sum, model.loss_sum,
                              model.L_GAN_sum])
    d_sum = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                              model.d_loss_fake_sum])
    writer = tf.summary.FileWriter(summary_dir, sess.graph)

    counter = iters+1
    start_time = time.time()

    while iters < num_iter:
        mini_batches = get_minibatches_idx(len(train_vids), batch_size, shuffle=True)
        for _, batchidx in mini_batches:
          if len(batchidx) == batch_size:

            t0 = time.time()
            Ts = np.repeat(np.array([T]),batch_size,axis=0)
            Ks = np.repeat(np.array([K]),batch_size,axis=0)
            seq_batch = load_kth_data_from_list(train_vids,batchidx,image_size,K,T,B)
            if debug:
                print seq_batch.shape
            seq_batch_tran = create_missing_frames(seq_batch.transpose([0,3,1,2,4]),K,T)
            forward_seq = seq_batch_tran
            if updateD:
              _, summary_str = sess.run([d_optim, d_sum],
                                         feed_dict={model.forward_seq: forward_seq,
                                                    model.target: seq_batch,
                                                    model.is_dis:True,
                                                    model.is_gen:False})
              writer.add_summary(summary_str, counter)

            if updateG:
              _, summary_str = sess.run([g_optim, g_sum],
                                        feed_dict={model.forward_seq: forward_seq,
                                                   model.target: seq_batch,
                                                    model.is_dis:False,
                                                    model.is_gen:True})
              writer.add_summary(summary_str, counter)

            errD_fake = model.d_loss_fake.eval({model.forward_seq: forward_seq,
                                                   model.target: seq_batch,
                                                    model.is_dis:False,
                                                    model.is_gen:False})
            errD_real = model.d_loss_real.eval({model.forward_seq: forward_seq,
                                                   model.target: seq_batch,
                                                    model.is_dis:False,
                                                    model.is_gen:False})
            errG = model.L_GAN.eval({model.forward_seq: forward_seq,
                                                   model.target: seq_batch,
                                                    model.is_dis:False,
                                                    model.is_gen:False})

            if errD_fake < margin or errD_real < margin:
              updateD = False
            if errD_fake > (1.-margin) or errD_real > (1.-margin):
              updateG = False
            if not updateD and not updateG:
              updateD = True
              updateG = True

            counter += 1

            print(
                "Iters: [%2d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f"
                % (iters, time.time() - start_time, errD_fake+errD_real,errG)
            )

            if np.mod(counter, 10000) == 1:
              seq_batch = load_kth_data_from_list(test_vids, range(batch_size), image_size, K, T)
              seq_batch_tran = seq_batch.transpose([0, 3, 1, 2, 4])
              forward_seq = seq_batch_tran
              samples = sess.run([model.G],
                                  feed_dict={model.forward_seq: forward_seq,
                                                   model.target: seq_batch,
                                                    model.is_dis:False,
                                                    model.is_gen:False})[0]
              for i in range(batch_size / 2):
                  sample = samples[i].swapaxes(0,2).swapaxes(1,2)
                  sbatch = seq_batch[i,:,:,:].swapaxes(0,2).swapaxes(1,2)
                  sample = np.concatenate((sample,sbatch), axis=0)
                  print("Saving sample ...")
                  save_images(sample[:,:,:,::-1], [2, T],
                              samples_dir+"train_%s_%s.png" % (iters, i))
            if np.mod(counter, 10000) == 2:
              model.save(sess, checkpoint_dir, counter)

            iters += 1

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.0001, help="Base Learning Rate")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=8, help="Mini-batch size")
  parser.add_argument("--alpha", type=float, dest="alpha",
                      default=1.0, help="Image loss weight")
  parser.add_argument("--beta", type=float, dest="beta",
                      default=0.02, help="GAN loss weight")
  parser.add_argument("--image_size", type=int, dest="image_size",
                      default=128, help="Mini-batch size")
  parser.add_argument("--K", type=int, dest="K",
                      default=1, help="Number of frames of gt per block")
  parser.add_argument("--T", type=int, dest="T",
                      default=3, help="Number of frames synthesized per block")
  parser.add_argument("--B", type=int, dest="B",
                      default=5, help="number of blocks")
  parser.add_argument("--convlstm_layer_num", type=int, dest="convlstm_layer_num",
                      default=3, help="number of convlstm layers")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=100000, help="Number of iterations")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", default="0",
                      help="GPU device id")
  parser.add_argument("--cpu", action="store_true", dest="cpu", help="use cpu only")
  parser.add_argument("--load_pretrain", action="store_true", dest="load_pretrain", help="load_pretrain")
  parser.add_argument("--tf_record_train_dir", type=str, nargs="?", dest="tf_record_train_dir",
                      default="../../../tf_record/KTH/train/", help="tf_record train location")
  parser.add_argument("--tf_record_test_dir", type=str, nargs="?", dest="tf_record_test_dir",
                      default="../../../tf_record/KTH/test/", help="tf_record test location")
  parser.add_argument("--color_channel_num", type=int, dest="color_channel_num",
                      default=1, help="number of color channels")
  parser.add_argument("--fea_enc_model", type=str, dest="fea_enc_model",default="pooling",
                    help="feature extraction model")
  parser.add_argument("--dyn_enc_model", type=str, dest="dyn_enc_model", default="mix",
                      help="dynamic encoding model")
  parser.add_argument("--reference_mode", type=str, dest="reference_mode", default="two",
                      help="refer to how many frames in the end")
  parser.add_argument("--debug", action="store_true", dest="debug", help="debug mode")

  args = parser.parse_args()
  main(**vars(args))
