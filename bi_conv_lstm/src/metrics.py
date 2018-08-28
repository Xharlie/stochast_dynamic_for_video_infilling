import numpy as np
import math
from scipy import signal

def cal_seq(gt, pred):
    # T is time dimension, gt[0] is batch, channel, shape, dimension like 60 * 64 * 64, bs is batch size
    bs = gt.shape[0]
    T = gt.shape[1]
    Channel = gt.shape[-1]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(Channel):
                res = cal_ssim(gt[i,t,:,:,c], pred[i,t,:,:,c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += cal_psnr(gt[i,t,:,:,c], pred[i,t,:,:,c])
            ssim[i, t] /= Channel
            psnr[i, t] /= Channel
            mse[i, t] = mse_metric(gt[i,t,...], pred[i,t,...])
    mean_psnr = np.mean(psnr, 1)
    mean_ssim = np.mean(ssim, 1)
    mean_batch_psnr = np.mean(mean_psnr)
    mean_batch_ssim = np.mean(mean_ssim)
    return mse, ssim, psnr, mean_ssim, mean_psnr, mean_batch_psnr, mean_batch_ssim

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def cal_psnr(x, y):
    mse = ((x - y)**2).mean()
    return 10*np.log(1/mse)/np.log(10)

def cal_ssim(img1, img2, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err