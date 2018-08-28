import os
import glob
import load_save_data


def load_dataset(opt):
    data_root = "../../../data/" + opt.dataset.upper()
    if opt.dataset == 'smmnist':
        num_digits = 2
        from moving_mnist import MovingMNIST
        train_data = MovingMNIST(
            train=True,
            data_root=data_root,
            seq_len = opt.B * (opt.K + opt.T) * 2,
            image_size=opt.image_size_h,
            deterministic=False,
            num_digits=num_digits)
        test_data = MovingMNIST(
            train=False,
            data_root=data_root,
            seq_len= opt.B * (opt.K + opt.T) * 2,
            image_size=opt.image_size_h,
            deterministic=False,
            num_digits=num_digits)
    elif opt.dataset == 'bair':
        from bair import RobotPush
        train_data = RobotPush(
            data_root= data_root,
            train=True,
            seq_len=30,
            image_size=opt.image_size_h)
        test_data = RobotPush(
            data_root= data_root,
            train=False,
            seq_len=30,
            image_size=opt.image_size_h)
    elif opt.dataset == 'kth':
        return load_dataset_raw(opt)
    elif opt.dataset == 'kth_sto':
        from kth_sto import KTH_STO
        train_data = KTH_STO(
            train=True,
            data_root= data_root,
            seq_len=opt.B * (opt.K + opt.T) * 2,
            image_size = opt.image_size_h)
        test_data = KTH_STO(
            train=False,
            data_root=data_root,
            seq_len=opt.B * (opt.K + opt.T) * 2,
            image_size=opt.image_size_h)

    return train_data, test_data

def check_create_dir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)
    return dir

def load_dataset_raw(args):
    tf_record_train_dir = "../../../tf_record/" + args.dataset.upper() + "_" + str(args.image_size_h) + "/train/"
    tf_record_test_dir = "../../../tf_record/" + args.dataset.upper() + "_" + str(args.image_size_h) + "/test/"
    check_create_dir(tf_record_train_dir)
    check_create_dir(tf_record_test_dir)
    train_tf_record_files = glob.glob(tf_record_train_dir + '*tfrecords*')
    test_tf_record_files = glob.glob(tf_record_test_dir + '*tfrecords*')

    train_vids = []
    if len(train_tf_record_files) == 0:
        print "len(train_tf_record_files) == 0:"
        data_path = "../../../data/KTH/"
        f = open(data_path + "train_data_list_trimmed.txt", "r")
        trainfiles = f.readlines()
        train_vids = load_save_data.save_data2record(
            trainfiles, data_path, args.image_size_h, args.image_size_w,
            tf_record_train_dir, args.color_channel_num)
    else:
        train_vids = load_save_data.load_records(glob.glob(tf_record_train_dir + '*tfrecords*'),
                                                 length=args.B*(args.K+args.T)+args.K)
    print "len(train_vids)", len(train_vids)
    test_vids = []
    if len(test_tf_record_files) == 0:
        data_path = "../../../data/KTH/"
        f = open(data_path + "test_data_list.txt", "r")
        testfiles = f.readlines()
        test_vids = load_save_data.save_data2record(
            testfiles, data_path, args.image_size_h, args.image_size_w,
            tf_record_test_dir, args.color_channel_num)
    else:
        test_vids = load_save_data.load_records(glob.glob(tf_record_test_dir + '*tfrecords*'))
    print "len(test_vids)", len(test_vids)
    return train_vids, test_vids