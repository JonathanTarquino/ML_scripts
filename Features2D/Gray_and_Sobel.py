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


def rangefilt(I,window):
  if I.dtype in ['uint8','uint16']:
    kernel= footprint_rectangle((window,window))
    imgOutput = gradient(I,kernel)
    #print('Rangefilt output',imgOutput)
  else:
    Im=I.astype(np.uint8)
    kernel= footprint_rectangle((window,window))
    imgOutput = gradient(Im,kernel)
    #print('Rangefilt output',imgOutput)

  return imgOutput


def grayfilts2(img,WindowSize=3):
  print(':::::::::::::::::::::::::')
  # grayFilt = []
  # grayFilt = pd.DataFrame(grayFilt)
  imShape = np.shape(img)
  if len(imShape)>2:
    print('Only 2D images supported, see GRAYFILTS3 otherwise.\n')
    return ()
  else:
    grayFilt = np.zeros((np.shape(img)[0],np.shape(img)[1],4))

    # Calculating Mean image
    print('Calculating Mean Image.\n')
    kernel = np.ones([WindowSize,WindowSize])/(WindowSize*WindowSize)
    filtResult = convolve2d(img, kernel, mode='same')
    grayFilt[:,:,0] = filtResult
    # grayFilt = pd.concat([grayFilt,pd.DataFrame(filtResult)])

    # Calculating Median image
    print('Calculating Median Image.\n')
    filtResult = medfilt2d(img,kernel_size=WindowSize)
    grayFilt[:,:,1] = filtResult
    # grayFilt = pd.concat([grayFilt,pd.DataFrame(filtResult)])

    # Calculating windowed standard deviation filter
    print('Calculating std Image.\n')
    filtResult = generic_filter(img, np.std, size=WindowSize)
    grayFilt[:,:,2] = filtResult
    # grayFilt = pd.concat([grayFilt,pd.DataFrame(filtResult)])

    # Calculating windowed range filter
    print('Calculating Windowed range image.\n')
    filtResult = rangefilt(img, WindowSize)
    grayFilt[:,:,3] = filtResult
    # grayFilt = pd.concat([grayFilt,pd.DataFrame(filtResult)])
    print(np.shape(grayFilt))
  return grayFilt

## -------------------------------------------------------------------------
def sobelxydiag(img):
  sobel = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
  Y = convolve2d(img, sobel, mode='same')
  return Y
## -------------------------------------------------------------------------

def sobelyxdiag(img):
  sobel = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
  sobel = np.fliplr(sobel)
  Y = convolve2d(img, sobel, mode='same')
  return Y

## -------------------------------------------------------------------------

def dx(img):
  Y = np.diff(img,axis=0)
  return Y

## -------------------------------------------------------------------------

def dy(img):
  mask = np.array([[1 ],[-1]])
  Y = convolve(img,mask,'same')
  return Y

## -------------------------------------------------------------------------

def ddiag(img):
  mask = np. array([[-1 ,0],[0,1]])
  Y = convolve2d(img,mask,'same')
  return Y

def gradfilts2(img):
  imSize = np.shape(img)
  if len(imSize)>2:
    print('Error: only 2D images are supported, provided:', imSize)
    return []

  else:
    feat_names = ['Gradient sobelx','Gradient sobely','Gradient sobelxy','Gradient sobelyx','Gradient x','Gradient y','Gradient magnitude','Gradient dx','Gradient dy','Gradient diagonal']

    nfeatures=len(feat_names);
    gradfeats=np.ones([imSize[0],imSize[1], nfeatures]);


    print('Calculating x,y Sobel edge images.\n')
    gradfeats[:,:,0] = sobel_h(img)
    gradfeats[:,:,1] = sobel_v(img)

    print('Calculating diagonal Sobel edge images.\n')
    gradfeats[:,:,2] = sobelxydiag(img)
    gradfeats[:,:,3] = sobelyxdiag(img)

    print('Calculating directional and magnitude gradients.\n')
    gradfeats[:,:,4],gradfeats[:,:,5] = np.gradient(img)

    gradfeats[:,:,6]=np.sqrt(gradfeats[:,:,4]**2 +gradfeats[:,:,5]**2)
    gradfeats[:,:,7]=np.vstack([dx(img),np.zeros(np.shape(img)[1])])
    gradfeats[:,:,8]=dy(img)
    gradfeats[:,:,9]=ddiag(img)

    return gradfeats
