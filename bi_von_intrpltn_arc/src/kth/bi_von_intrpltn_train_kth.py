import sys
import time
import tensorflow as tf
import glob

from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed
sys.path.append("../")
import load_save_data
from bi_von_model import bi_von_net
from utils import *

def main(lr, batch_size, alpha, beta, image_size, K,
         T, num_iter, gpu, tf_record_dir, color_channel_num, cpu):
  tf_record_files=glob.glob(tf_record_dir+'*.tfrecords')
  vids = []
  if len(tf_record_files) == 0:
      data_path = "../../../data/KTH/"
      f = open(data_path+"train_data_list_trimmed.txt","r")
      trainfiles = f.readlines()
      vids = load_save_data.save_kth_data2record(
          trainfiles, data_path, image_size, tf_record_dir, color_channel_num)
  else:
      vids = load_save_data.load_kth_records(tf_record_dir)
  margin = 0.3
  updateD = True
  updateG = True
  iters = 0
  prefix  = ("KTH_lstr"
          + "_image_size="+str(image_size)
          + "_K="+str(K)
          + "_T="+str(T)
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

  device_string="/gpu:%d"%gpu[0]
  if cpu:
      device_string = "/cpu:0"
  with tf.device(device_string):
    model = bi_von_net(image_size=[image_size,image_size], c_dim=1,
                  K=K, batch_size=batch_size, T=T,
                  checkpoint_dir=checkpoint_dir)
    # for var in tf.trainable_variables():
    #     print var.name
    # raw_input('print trainable variables...')
    d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        model.d_loss, var_list=model.d_vars
    )
    g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        alpha*model.L_img+beta*model.L_GAN, var_list=model.g_vars
    )

  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
  config=tf.ConfigProto(allow_soft_placement=True,
                  log_device_placement=False)
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:

    tf.global_variables_initializer().run()

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
        mini_batches = get_minibatches_idx(len(vids), batch_size, shuffle=True)
        for _, batchidx in mini_batches:
          if len(batchidx) == batch_size:
            seq_batch = np.zeros((batch_size, image_size, image_size,
                                   2*K+T, 1), dtype="float32")
            t0 = time.time()
            Ts = np.repeat(np.array([T]),batch_size,axis=0)
            Ks = np.repeat(np.array([K]),batch_size,axis=0)
            # paths = np.repeat(data_path, batch_size,axis=0)
            # tfiles = np.array(trainfiles)[batchidx]
            # shapes = np.repeat(np.array([image_size]),batch_size,axis=0)
            # output = parallel(delayed(load_kth_data)(f, p,img_sze, k, t)
            #                                      for f,p,img_sze,k,t in zip(tfiles,
            #                                                                 paths,
            #                                                                 shapes,
            #                                                                 Ks, Ts))
            # for i in xrange(batch_size):
            #   seq_batch[i] = output[i]
            seq_batch = vids[batchidx]
            seq_batch_tran = seq_batch.transpose([0,3,1,2,4])
            forward_seq = seq_batch_tran[:,:K,:,:,:]
            backward_seq = seq_batch_tran[:,::-1][:,:K,:,:,:]
            if updateD:
              _, summary_str = sess.run([d_optim, d_sum],
                                         feed_dict={model.forward_seq: forward_seq,
                                                    model.backward_seq: backward_seq,
                                                    model.fxt: seq_batch[:,:,:,K-1,:],
                                                    model.bxt: seq_batch[:,:,:,T+K-1,:],
                                                    model.target: seq_batch})
              writer.add_summary(summary_str, counter)

            if updateG:
              _, summary_str = sess.run([g_optim, g_sum],
                                        feed_dict={model.forward_seq: forward_seq,
                                                   model.backward_seq: backward_seq,
                                                   model.fxt: seq_batch[:, :, :, K - 1, :],
                                                   model.bxt: seq_batch[:, :, :, T + K - 1, :],
                                                   model.target: seq_batch})
              writer.add_summary(summary_str, counter)

            errD_fake = model.d_loss_fake.eval({model.forward_seq: forward_seq,
                                                   model.backward_seq: backward_seq,
                                                   model.fxt: seq_batch[:, :, :, K - 1, :],
                                                   model.bxt: seq_batch[:, :, :, T + K - 1, :],
                                                   model.target: seq_batch})
            errD_real = model.d_loss_real.eval({model.forward_seq: forward_seq,
                                                   model.backward_seq: backward_seq,
                                                   model.fxt: seq_batch[:, :, :, K - 1, :],
                                                   model.bxt: seq_batch[:, :, :, T + K - 1, :],
                                                   model.target: seq_batch})
            errG = model.L_GAN.eval({model.forward_seq: forward_seq,
                                                   model.backward_seq: backward_seq,
                                                   model.fxt: seq_batch[:, :, :, K - 1, :],
                                                   model.bxt: seq_batch[:, :, :, T + K - 1, :],
                                                   model.target: seq_batch})

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

            if np.mod(counter, 100) == 1:
              samples = sess.run([model.G],
                                  feed_dict={model.forward_seq: forward_seq,
                                                   model.backward_seq: backward_seq,
                                                   model.fxt: seq_batch[:, :, :, K - 1, :],
                                                   model.bxt: seq_batch[:, :, :, T + K - 1, :],
                                                   model.target: seq_batch})[0]
              samples = samples[0].swapaxes(0,2).swapaxes(1,2)
              sbatch  = seq_batch[0,:,:,K:K+T].swapaxes(0,2).swapaxes(1,2)
              samples = np.concatenate((samples,sbatch), axis=0)
              print("Saving sample ...")
              save_images(samples[:,:,:,::-1], [2, T],
                          samples_dir+"train_%s.png" % (iters))
            if np.mod(counter, 500) == 2:
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
                      default=10, help="Number of steps to observe from the past")
  parser.add_argument("--T", type=int, dest="T",
                      default=10, help="Number of steps into the future")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=100000, help="Number of iterations")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", default="0",
                      help="GPU device id")
  parser.add_argument("--cpu", action="store_true", dest="cpu", help="use cpu only")
  parser.add_argument("--tf_record_dir", type=str, nargs="?", dest="tf_record_dir",
                      default="../../../tf_record/KTH/", help="tf_record location")
  parser.add_argument("--color_channel_num", type=int, dest="color_channel_num",
                      default=1, help="number of color channels")
  args = parser.parse_args()
  main(**vars(args))
