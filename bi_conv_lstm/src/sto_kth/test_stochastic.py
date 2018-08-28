import sys
import ssim
import skimage.measure as measure
from PIL import Image
sys.path.append("../")
sys.path.append("../../../data/stochastic_dataset_script/")
import sys
import tensorflow as tf
from os import listdir, makedirs, system
from argparse import ArgumentParser
sys.path.append("../")
sys.path.append("../../../data/stochastic_dataset_script/")
from stochastic_bi_model import stochastic_bi_net
from stochastic_learned_prior import stochastic_learned_prior
from deter_flexible import deter_flexible
import data_util
from utils import *
import ops

parser = ArgumentParser()
parser.add_argument("--lr", type=float, dest="lr",
                    default=0.002, help="Base Learning Rate")
parser.add_argument("--batch_size", type=int, dest="batch_size",
                    default=8, help="Mini-batch size")
parser.add_argument("--beta1", type=float, dest="beta1",
                    default=0.5, help="beta1 of adams")
parser.add_argument("--alpha", type=float, dest="alpha",
                    default=0.0001, help="inclusion KL weight")
parser.add_argument("--gamma", type=float, dest="gamma",
                    default=0.0, help="exclusion KL weight")
parser.add_argument("--beta", type=float, dest="beta",
                    default=0.4, help="posterior and inference weight")
parser.add_argument("--z_channel", type=int, dest="z_channel",
                    default=24, help="Image loss weight")
parser.add_argument("--decay_step", type=float, dest="decay_step",
                    default=40000)
parser.add_argument("--decay_rate", type=float, dest="decay_rate",
                    default=0.9)
parser.add_argument("--image_size_h", type=int, dest="image_size_h",
                    default=64, help="Mini-batch size")
parser.add_argument("--image_size_w", type=int, dest="image_size_w",
                    default=64, help="Mini-batch size")
parser.add_argument("--K", type=int, dest="K",
                    default=1, help="Number of frames of gt per block")
parser.add_argument("--T", type=int, dest="T",
                    default=3, help="Number of frames synthesized per block")
parser.add_argument("--B", type=int, dest="B",
                    default=5, help="number of blocks")
parser.add_argument("--num_iter", type=int, dest="num_iter",
                    default=120001, help="Number of iterations")
parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", default=0,
                    help="GPU device id")
parser.add_argument("--cpu", action="store_true", dest="cpu", help="use cpu only")
parser.add_argument("--no_normalized", action="store_true", dest="no_normalized", help="no_normalized")
parser.add_argument("--mask_weight", type=int, dest="mask_weight", default=3, help="mask_weight")
parser.add_argument("--start_percentage", type=float, dest="start_percentage",
                    default=1.0, help="start_percentage")
parser.add_argument("--color_channel_num", type=int, dest="color_channel_num",
                    default=1, help="number of color channels")
parser.add_argument("--debug", action="store_true", dest="debug", help="debug mode")
parser.add_argument("--no_store", action="store_true", dest="no_store", help="no_store")
parser.add_argument("--no_res_ref", action="store_true", dest="no_res_ref", help="no_res_ref")
parser.add_argument("--no_negative_noise", action="store_true", dest="no_negative_noise", help="no_negative_noise")
parser.add_argument("--no_space_aware", action="store_true", dest="no_space_aware", help="space_aware")
parser.add_argument("--no_pic_norm", action="store_true", dest="no_pic_norm", help="no_pic_norm")
parser.add_argument("--iters", type=int, dest="iters", default=0)
parser.add_argument("--mode", type=str, dest="mode",
                        default="bi_sto", help="number frames to be sent in discriminator")
parser.add_argument("--cell_type", type=str, dest="cell_type",
                        default="gru", help="lstm or gru")
parser.add_argument("--pretrain_model", type=str, dest="pretrain_model",
                    default="", help="")
parser.add_argument("--pixel_loss", type=str, dest="pixel_loss",
                    default="l1", help="")
parser.add_argument("--res_type", type=str, dest="res_type",
                    default="avg", help="")
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument("--quantitative", action="store_true", dest="quantitative", help="quantitative")
parser.add_argument("--qualitative", action="store_true", dest="qualitative", help="qualitative")
parser.add_argument("--gif_per_vid", type=int, dest="gif_per_vid", default=1,
                    help="refer to how many per video")
parser.add_argument("--save_gt", action="store_true", dest="save_gt", help="no_store")
parser.add_argument("--best_model", type=str, dest="best_model", default="", help="")
parser.add_argument("--testtrain", action="store_true", dest="testtrain", help="no_store")
args = parser.parse_args()
space_aware = not args.no_space_aware

def image_clipping(img):
    img = inverse_transform(img) * 255
    img = np.clip(img, 0, 255, out=img)
    return img.astype(np.uint8)

def write2gt(gt_dir, test_vids, flipable, c_dim=1):
    ind=0
    while True:
        batch_lst = range(args.batch_size)
        if args.dataset == "kth":
            if ind + ind + args.batch_size >= len(test_vids):
                break
            batch_lst = range(ind, ind + args.batch_size)
        ref_batch_test, inf_batch_test = load_stochastic_data_from_list(
            test_vids, batch_lst,
            args.image_size_h, args.image_size_w, args.K, args.T, args.B, flipable=flipable,
            channel=args.color_channel_num)
        if ref_batch_test is None:
            break
        ref_batch_test = (inverse_transform(ref_batch_test) * 255).astype("uint8")
        inf_batch_test = (inverse_transform(inf_batch_test) * 255).astype("uint8")
        for i in range(ref_batch_test.shape[0]):
            ref_seq = ref_batch_test[i,...]
            inf_seq = inf_batch_test[i,...]
            img_dir = gt_dir+str(ind)+"/"
            check_create_dir(img_dir)
            for t in range(ref_seq.shape[0]):
                if c_dim == 1:
                    cv2.imwrite(img_dir + "ref_" + "{0:04d}".format(t+1) + ".png", ref_seq[t,...])
                else:
                    cv2.imwrite(img_dir + "ref_" + "{0:04d}".format(t + 1) + ".png", cv2.cvtColor(ref_seq[t, ...], cv2.COLOR_RGB2BGR))
            for t in range(inf_seq.shape[0]):
                if c_dim == 1:
                    cv2.imwrite(img_dir + "gt_" + "{0:04d}".format(t + 1) + ".png", inf_seq[t, ...])
                else:
                    cv2.imwrite(img_dir + "gt_" + "{0:04d}".format(t + 1) + ".png", cv2.cvtColor(inf_seq[t, ...], cv2.COLOR_RGB2BGR))
            cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + img_dir +
                    "gt_%04d.png " + img_dir + "gt.gif")
            print cmd2
            os.system(cmd2)
            ind+=args.batch_size

def main():

    train_vids, test_vids = data_util.load_dataset(args)

    iters = args.iters
    best_model = args.best_model  # will pick last model
    prefix = ("sto"
              + "_h=" + str(args.image_size_h)
              + "_w=" + str(args.image_size_w)
              + "_K=" + str(args.K)
              + "_T=" + str(args.T)
              + "_B=" + str(args.B)
              + "_batch_size=" + str(32)
              + "_beta1=" + str(args.beta1)
              + "_alpha=" + str(args.alpha)
              + "_gamma=" + str(args.gamma)
              + "_lr=" + str(args.lr)
              + "_mode=" + str(args.mode)
              + "_space_aware=" + str(space_aware)
              + "_z_channel=" + str(args.z_channel)
              + "_p_loss=" + str(args.pixel_loss)
              + "_cell_type=" + str(args.cell_type)
              + "_norm=" + str(not args.no_normalized)
              + "_mask_w=" + str(args.mask_weight)
              + "_res_type=" + str(args.res_type)
              + "_neg_noise=" + str(not args.no_negative_noise)
              + "_res_ref=" + str(not args.no_res_ref)
              + "_pic_norm=" + str(not args.no_pic_norm)
              + "_start_perc=" + str(args.start_percentage)
              )

    checkpoint_dir = "../../models/stochastic/" \
                     + args.dataset + '/' + prefix + "/"
    # if args.best_model!="":
    #     prefix+="_" + args.best_model
    device_string = ""
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device_string = "/cpu:0"
    elif args.gpu:
        device_string = "/gpu:%d" % args.gpu[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu[0])
    gt_dir = "../../results/images/stochastic/" + args.dataset + '/' + str(args.T) + "/gt/"
    c_dim = args.color_channel_num
    flipable = False
    if args.dataset=="kth":
        flipable = True
    if args.save_gt:
        write2gt(gt_dir, test_vids, flipable, c_dim=c_dim)
        return
    with tf.device(device_string):
        if args.mode == "bi_sto":
            model = stochastic_bi_net([args.image_size_h, args.image_size_w], batch_size = args.batch_size,
                   c_dim = args.color_channel_num, K=args.K, T=args.T, B=args.B, debug = False,
                   pixel_loss = args.pixel_loss, convlstm_kernel = [3, 3], mode = args.mode,
                   space_aware = space_aware, cell_type=args.cell_type, z_channel = args.z_channel,
                   normalize = not args.no_normalized, weight=args.mask_weight, res_type=args.res_type,
                   negative_noise = not args.no_negative_noise, res_ref = not args.no_res_ref,
                   pic_norm = not args.no_pic_norm)
        elif args.mode == "learned_prior":
            model = stochastic_learned_prior([args.image_size_h, args.image_size_w], batch_size = args.batch_size,
                       c_dim = args.color_channel_num, K=args.K, T=args.T, B=args.B, debug = False,
                       pixel_loss = args.pixel_loss, convlstm_kernel = [3, 3], mode = args.mode,
                       space_aware = space_aware, cell_type=args.cell_type, z_channel = args.z_channel,
                       normalize = not args.no_normalized, weight=args.mask_weight, res_type=args.res_type,
                       negative_noise = not args.no_negative_noise, res_ref = not args.no_res_ref,
                       pic_norm = not args.no_pic_norm)
        elif args.mode == "deter_flexible":
            model = deter_flexible([args.image_size_h, args.image_size_w], batch_size = args.batch_size,
                       c_dim = args.color_channel_num, K=args.K, T=args.T, B=args.B, debug = False,
                       pixel_loss = args.pixel_loss, convlstm_kernel = [3, 3], mode = args.mode,
                       space_aware = space_aware, cell_type=args.cell_type, z_channel = args.z_channel,
                       normalize = not args.no_normalized, weight=args.mask_weight, res_type=args.res_type,
                       negative_noise = not args.no_negative_noise, res_ref = not args.no_res_ref,
                       pic_norm = not args.no_pic_norm)

        # global_step = tf.Variable(0, trainable=False)
        # global_rate = tf.train.exponential_decay(args.lr, global_step,
        #              args.decay_step, args.decay_rate, staircase=True)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     g_full = model.L_train_p + args.alpha * model.L_train_kl
        #     if args.gamma != 0:
        #         g_full += args.gamma * model.L_train_kl_exlusive
        #     g_optim = tf.train.AdamOptimizer(global_rate, beta1=args.beta1).minimize(
        #         g_full, var_list=model.trainable_variables, global_step=global_step
        #     )
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        print("checkpoint_dir:",checkpoint_dir)
        loaded, model_name = ops.load(model, sess, checkpoint_dir, best_model)
        gen_dir = "../../results/images/stochastic/" + args.dataset + '/' + str(args.T) + "/" \
                  + prefix + "/generated/" + model_name + "/"
        quant_dir = "../../results/quantitative/stochastic/" + args.dataset + '/' + str(
            args.T) + "/" + prefix + "/quant/" + model_name + "/"
        if args.save_gt:
            check_create_dir(gt_dir, clean=True)
        if not args.save_gt and args.qualitative:
            check_create_dir(gen_dir, clean=False)
        if not args.save_gt and args.quantitative:
            check_create_dir(quant_dir, clean=True)

        save_path = quant_dir + "results_model=" + model_name + ".npz"
        save_path_post = quant_dir + "results_model=" + model_name + "_post.npz"
        save_path_one = quant_dir + "results_model=" + model_name + "_one.npz"
        p_loss_percentage = 1.0
        psnr_err = np.zeros((0, args.T))
        flow_err = np.zeros((0, 1))
        ssim_err = np.zeros((0, args.T))
        psnr_err_post = np.zeros((0, args.T))
        flow_err_post = np.zeros((0, 1))
        ssim_err_post = np.zeros((0, args.T))

        for img_dir in subdir(gt_dir):
            gensub_dir = gen_dir +  img_dir.split('/')[-1] + "/"
            check_create_dir(gensub_dir, clean=(not args.testtrain))

            inf_batch = np.zeros((1, args.K + args.T + 1, args.image_size_h, args.image_size_w,
                                  c_dim), dtype="float32")
            ref_batch = np.zeros((1, args.B + 1, args.image_size_h, args.image_size_w, c_dim), dtype="float32")
            for t in range(args.B + 1):
                img = cv2.imread(img_dir + "/ref_" + "{0:04d}".format(t+1) + ".png")
                if c_dim == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if c_dim == 1:
                    ref_batch[0, t, ...] = transform(img[:, :, None])
                else:
                    ref_batch[0, t, ...] = transform(img[:, :, :])
            for t in range(args.K + args.T + 1):
                img = cv2.imread(img_dir + "/gt_" + "{0:04d}".format(t + 1) + ".png")
                if c_dim == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if c_dim == 1:
                    inf_batch[0, t, ...] = transform(img[:, :, None])
                else:
                    inf_batch[0, t, ...] = transform(img[:, :, :])

            true_data = inf_batch.copy()
            pred_data = np.zeros((1, args.K + args.T + 1, args.image_size_h, args.image_size_w,
                                  c_dim), dtype="float32")
            pred_data_post = np.zeros((1, args.K + args.T + 1, args.image_size_h, args.image_size_w,
                                  c_dim), dtype="float32")
            G_test, G_test_post, test_mask_binary = sess.run(
                [model.G_real, model.G_post_real, model.mask_binary],
                feed_dict={model.ref_seq: ref_batch,
                           model.inf_seq: inf_batch,
                           model.is_train: args.testtrain,
                           model.p_loss_percentage: p_loss_percentage})
            print G_test.shape, inf_batch[:,0,...].shape
            pred_data[0] = np.concatenate((np.expand_dims(inf_batch[:,0,...],axis=1), G_test,
                                           np.expand_dims(inf_batch[:,-1,...],axis=1)),axis=1)
            pred_data_post[0] = np.concatenate((np.expand_dims(inf_batch[:, 0, ...], axis=1), G_test_post,
                                                np.expand_dims(inf_batch[:, -1, ...], axis=1)), axis=1)
            true_data_unit = normalized2uint(true_data)
            pred_data_unit = normalized2uint(pred_data)
            pred_data_post_unit = normalized2uint(pred_data_post)
            cpsnr = np.zeros((args.T))
            cssim = np.zeros((args.T))
            cpsnr_post = np.zeros((args.T))
            cssim_post = np.zeros((args.T))
            flow_l2 = np.zeros((1,1))
            flow_l2_post = np.zeros((1,1))
            if args.quantitative:
                for t in xrange(args.T):
                    pred = pred_data_unit[0, t+args.K, ...]
                    pred_post = pred_data_post_unit[0, t+args.K, ...]
                    target = true_data_unit[0, t+args.K, ...]
                    cpsnr[t] = measure.compare_psnr(target, pred)
                    cssim[t] = ssim.compute_ssim(Image.fromarray(cv2.cvtColor(target,
                                                                              cv2.COLOR_GRAY2BGR if c_dim == 1 else cv2.COLOR_RGB2BGR)),
                                                 Image.fromarray(cv2.cvtColor(pred,
                                                                              cv2.COLOR_GRAY2BGR if c_dim == 1 else cv2.COLOR_RGB2BGR)))
                    cpsnr_post[t] = measure.compare_psnr(target, pred_post)
                    cssim_post[t] = ssim.compute_ssim(Image.fromarray(cv2.cvtColor(target,
                                                                                   cv2.COLOR_GRAY2BGR if c_dim == 1 else cv2.COLOR_RGB2BGR)),
                                                 Image.fromarray(cv2.cvtColor(pred_post,
                                                                                    cv2.COLOR_GRAY2BGR if c_dim == 1 else cv2.COLOR_RGB2BGR)))
                flow_target = cv2.calcOpticalFlowFarneback(
                    true_data_unit[0, args.T + args.K - 1, ...] if c_dim == 1 else cv2.cvtColor(true_data_unit[0, args.T + args.K - 1, ...], cv2.COLOR_RGB2GRAY),
                    true_data_unit[0, args.T + args.K, ...] if c_dim == 1 else cv2.cvtColor(true_data_unit[0, args.T + args.K, ...], cv2.COLOR_RGB2GRAY)
                                                                                            , 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_pred = cv2.calcOpticalFlowFarneback(
                    pred_data_unit[0, args.T + args.K - 1, ...] if c_dim == 1 else cv2.cvtColor(pred_data_unit[0, args.T + args.K - 1, ...], cv2.COLOR_RGB2GRAY),
                    pred_data_unit[0, args.T + args.K, ...] if c_dim == 1 else cv2.cvtColor(pred_data_unit[0, args.T + args.K, ...], cv2.COLOR_RGB2GRAY)
                    , 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_pred_post = cv2.calcOpticalFlowFarneback(
                    pred_data_post_unit[0, args.T + args.K - 1, ...] if c_dim == 1 else cv2.cvtColor(pred_data_post_unit[0, args.T + args.K - 1, ...], cv2.COLOR_RGB2GRAY),
                    pred_data_post_unit[0, args.T + args.K, ...] if c_dim == 1 else cv2.cvtColor(pred_data_post_unit[0, args.T + args.K, ...], cv2.COLOR_RGB2GRAY),
                    0.5, 3, 15, 3, 5, 1.2, 0)
                flow_l2[0,0] = np.mean(np.square(flow_target - flow_pred))
                flow_l2_post[0,0] = np.mean(np.square(flow_target - flow_pred_post))
            if args.qualitative:
                for t in xrange(args.K * 2 + args.T):
                        pred_frame = draw_frame(cv2.cvtColor(pred_data_unit[0, t, ...],
                                            cv2.COLOR_GRAY2BGR) if c_dim == 1 else pred_data_unit[0, t, ...],
                                                t % (args.T + args.K) < args.K)
                        pred_post_frame = draw_frame(cv2.cvtColor(pred_data_post_unit[0, t, ...]
                                          , cv2.COLOR_GRAY2BGR) if c_dim == 1 else pred_data_post_unit[0, t, ...], t % (args.T + args.K) < args.K)
                        if args.testtrain:
                            cv2.imwrite(gensub_dir + "predone_" + "{0:04d}".format(t) + ".png", pred_data_unit[0, t, ...])
                            cv2.imwrite(gensub_dir + "predoneframe_" + "{0:04d}".format(t) + ".png", pred_frame)
                        else:
                            cv2.imwrite(gensub_dir + "pred_" + "{0:04d}".format(t) + ".png", pred_data_unit[0, t, ...])
                            cv2.imwrite(gensub_dir + "predframe_" + "{0:04d}".format(t) + ".png", pred_frame)
                            cv2.imwrite(gensub_dir + "predpost_" + "{0:04d}".format(t) + ".png", pred_data_post_unit[0, t, ...])
                            cv2.imwrite(gensub_dir + "predpostframe_" + "{0:04d}".format(t) + ".png", pred_post_frame)
                        # blank = (inverse_transform(inf_batch[0, t, :, :]) * 255).astype("uint8")
                        # cv2.imwrite(savedir + "/blk_gt_" + "{0:04d}".format(t) + ".png", blank)
            if args.qualitative:
                cmd1 = "rm " + gensub_dir + "predframe.gif"
                cmd4 = "rm " + gensub_dir + "predpostframe.gif"
                cmd7 = "rm " + gensub_dir + "predoneframe.gif"
                cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + gensub_dir +
                        "predframe_%04d.png " + gensub_dir + "predframe.gif")
                cmd5 = ("ffmpeg -f image2 -framerate 7 -i " + gensub_dir +
                        "predpostframe_%04d.png " + gensub_dir + "predpostframe.gif")
                cmd8 = ("ffmpeg -f image2 -framerate 7 -i " + gensub_dir +
                        "predoneframe_%04d.png " + gensub_dir + "predoneframe.gif")
                cmd3 = "rm " + gensub_dir + "predframe*.png"
                cmd6 = "rm " + gensub_dir + "predpostframe*.png"
                cmd9 = "rm " + gensub_dir + "predoneframe*.png"

                # Comment out "system(cmd3)" if you want to keep the output images
                # Otherwise only the gifs will be kept
                if args.testtrain:
                    system(cmd7);
                    system(cmd8);
                    system(cmd9)
                else:
                    system(cmd1);
                    system(cmd2); system(cmd3);
                    system(cmd4);
                    system(cmd5); system(cmd6);
            if args.quantitative:
                print psnr_err.shape, cpsnr.shape
                print ssim_err.shape, cssim.shape
                print "ssim_err of this sequence", np.mean(cssim)
                print "ssim_err_post of this sequence", np.mean(cssim_post)
                print "psnr_err of this sequence", np.mean(cpsnr)
                print "psnr_err_post of this sequence", np.mean(cpsnr_post)
                psnr_err = np.concatenate((psnr_err, cpsnr[None,:]), axis=0)
                ssim_err = np.concatenate((ssim_err, cssim[None,:]), axis=0)
                flow_err = np.concatenate((flow_err, flow_l2[:]), axis=0)
                psnr_err_post = np.concatenate((psnr_err_post, cpsnr_post[None,:]), axis=0)
                ssim_err_post = np.concatenate((ssim_err_post, cssim_post[None,:]), axis=0)
                flow_err_post = np.concatenate((flow_err_post, flow_l2_post[:]), axis=0)

        if args.quantitative:
            if args.testtrain:
                np.savez(save_path_one, psnr=psnr_err, ssim=ssim_err, flow=flow_err)
            else:
                np.savez(save_path, psnr=psnr_err, ssim=ssim_err, flow=flow_err)
                np.savez(save_path_post, psnr=psnr_err_post, ssim=ssim_err_post , flow=flow_err_post)
        if args.testtrain:
            print("PriorOne Results saved to " + save_path)
            print "PriorOne PSNR per frame:", np.mean(psnr_err, axis=0)
            print "PriorOne SSIM per frame:", np.mean(ssim_err, axis=0)
            print "PriorOne PSNR overall average:", np.mean(psnr_err), "PriorOne SSIM overall average", \
                np.mean(ssim_err), "PriorOne flow_err average", np.mean(flow_err)
        else:
            print("Prior Results saved to " + save_path)
            print("Post Results saved to " + save_path_post)
            print "Prior PSNR per frame:", np.mean(psnr_err, axis = 0)
            print "Prior SSIM per frame:", np.mean(ssim_err, axis = 0)
            print "Prior PSNR overall average:", np.mean(psnr_err), "Prior SSIM overall average", \
                np.mean(ssim_err), "Prior flow_err average", np.mean(flow_err)
            print "Post PSNR per frame:", np.mean(psnr_err_post, axis = 0)
            print "Post SSIM per frame:", np.mean(ssim_err_post, axis = 0)
            print "Post PSNR overall average:", np.mean(psnr_err_post), "Post SSIM overall average",\
                np.mean(ssim_err_post), "Post flow_err average", np.mean(flow_err_post)
    print("Done.")

if __name__ == "__main__":
    main()
