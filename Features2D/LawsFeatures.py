from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
import math
import glob
import skimage
from scipy.signal import convolve2d, medfilt2d, convolve
from scipy.ndimage.filters import generic_filter
from skimage.filters import sobel_h, sobel_v, sobel
from skimage.filters.rank import gradient,mean
from skimage.morphology import erosion,dilation, footprint_rectangle
from skimage.util import img_as_uint, img_as_ubyte
import cv2 as cv
import mahotas
from scipy.stats import kurtosis, skew
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float

from skimage.filters import gabor_kernel
from pyfeats import lte_measures


def _image_xor(f):
    # Turn "0" to "1" and vice versa: XOR with image consisting of "1"s
    f = f.astype(np.uint8)
    mask = np.ones(f.shape, np.uint8)
    out = np.zeros(f.shape, np.uint8)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            out[i,j] = f[i,j] ^ mask[i,j]
    return out



import numpy as np
from scipy import signal
import warnings

def lte_measures(f, mask=None, l=7):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    l : int, optional
        Law's mask size. The default is 7.

    Returns
    -------
    features : numpy ndarray
        1)texture energy from LL kernel, 2) texture energy from EE
        kernel, 3)texture energy from SS kernel, 4)average texture
        energy from LE and EL kernels, 5)average texture energy from
        ES and SE kernels, 6)average texture energy from LS and SL
        kernels.
    labels : list
        Labels of features.
    '''
    # print('LAWS MEASSUre')
    if mask is None:
        mask = np.ones(f.shape)
        # print('----------',np.shape(mask))

    # 1) Labels
    labels = ["LTE_LL","LTE_LE","LTE_LS","LTE_EL","LTE_EE","LTE_ES","LTE_SL","LTE_SE","LTE_SS"]
    labels = [label+'_w_'+str(l) for label in labels]

    # 2) Parameters
    f = np.array(f, np.double)
    mask = np.array(mask, np.double)
    kernels = np.zeros((l,l,9), np.double)

    # 3) From 3 kernels [L, E, S], get 9 [LL, LE, LS, EL, EE, ES, SL, SE, SS]
    if l not in [3,5,7]:
        warnings.warn('Accepted vsize for Laws mask are 3, 5 and 7. Using 7 by default')
        l = 7
    if l==3:
        L = np.array([ 1,  2,  1], np.double)
        E = np.array([-1,  0,  1], np.double)
        S = np.array([-1,  2, -1], np.double)
    elif l==5:
        L = np.array([ 1,  4,  6,  4,  1], np.double)
        E = np.array([-1, -2,  0,  2,  1], np.double)
        S = np.array([-1,  0,  2,  0, -1], np.double)
    elif l==7:
        L = np.array([ 1,  6,  15,  20,  15,  6,  1], np.double)
        E = np.array([-1, -4,  -5,   0,   5,  4,  1], np.double)
        S = np.array([-1, -2,   1,   4,   1, -2, -1], np.double)

    oneskernel = np.ones((l,l), np.double)
    kernels = np.zeros((l,l,9), np.double)
    kernels[:,:,0] = np.multiply(L.reshape(-1,1),L) # LL kernel
    kernels[:,:,1] = np.multiply(L.reshape(-1,1),E) # LE kernel
    kernels[:,:,2] = np.multiply(L.reshape(-1,1),S) # LS kernel
    kernels[:,:,3] = np.multiply(E.reshape(-1,1),L) # EL kernel
    kernels[:,:,4] = np.multiply(E.reshape(-1,1),E) # EE kernel
    kernels[:,:,5] = np.multiply(E.reshape(-1,1),S) # ES kernel
    kernels[:,:,6] = np.multiply(S.reshape(-1,1),L) # SL kernel
    kernels[:,:,7] = np.multiply(S.reshape(-1,1),E) # SE kernel
    kernels[:,:,8] = np.multiply(S.reshape(-1,1),S) # SS kernel

    # 4) Get mask where convolution should be performed
    mask_c = _image_xor(mask)
    mask_conv = signal.convolve2d(mask_c, oneskernel,mode = 'same')
    mask_conv = np.abs(np.sign(mask_conv)-1)
    # print(np.shape(mask_conv))

    # 5) Calculate energy of each convolved image with each kernel: total 9
    energy = np.zeros((np.shape(mask_conv)[0],np.shape(mask_conv)[1],9),np.double)
    print('___________________________________________________________',energy.shape)
    area = sum(sum(mask_conv))
    for i in range(9):
        f_conv = signal.convolve2d(f, kernels[:,:,i], mode = 'same')
        f_conv = np.multiply(f_conv,mask_conv)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',np.shape(f_conv))
        energy[:,:,i] = f_conv
        # f_conv_mean = sum(sum(f_conv)) / area
        # energy[i] = np.sqrt(sum(sum(np.multiply((f_conv-f_conv_mean)**2,mask_conv)))/area)

    # 6) Calculate features
    # features = np.zeros(6,np.double)
    # features[0] = energy[0]
    # features[1] = energy[4]
    # features[2] = energy[8]
    # features[3] = (energy[1]+energy[3])/2
    # features[4] = (energy[5]+energy[7])/2
    # features[5] = (energy[2]+energy[6])/2

    print(energy.shape)
    return energy, labels
