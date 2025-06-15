#from radiomics import featureextractor
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



def compute_gabor(image, kernels):
    feats = np.zeros((np.shape(image)[0],np.shape(image)[1],len(kernels)), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[:,:,k] = filtered
    return feats

def gaborFilter(img):
  # preparing filter bank kernels
  gabor_names = []
  kernels = []
  for theta in range(7):
      th = theta / 8.0 * np.pi
      for frequency in (0.7653,1.2755,1.7857,2.2959,2.8061):
          kernel = np.real(gabor_kernel(frequency, theta=th))
          kernels.append(kernel)
          gabor_names.append(['GaborXY_theta= pi*'+str(theta)+'/8_lambda='+str(frequency)])
  print('Number of kernels:',len(kernels))

  ima = img_as_float(img)
  gabor_feats = compute_gabor(ima, kernels)
  print('------------------------------------',np.shape(gabor_feats),gabor_names)
  return(gabor_feats,gabor_names)
