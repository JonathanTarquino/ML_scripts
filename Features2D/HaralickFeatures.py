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


def extractHaralick(img,window,d=1):

  paddedImage = np.zeros((np.shape(img)[0]+(2*window),np.shape(img)[1]+(2*window)))
  print('Padded image shape:',np.shape(img),np.shape(paddedImage))
  HM = np.zeros((np.shape(img)[0],np.shape(img)[1],13))

  if img.dtype in ['uint8','uint16']:
    print('Making a int8 copy of the image')
    paddedImage[0+math.floor(window/2):np.shape(img)[0]+math.floor(window/2),0+math.floor(window/2):np.shape(img)[1]+math.floor(window/2)] = img
    paddedImage=paddedImage.astype(np.uint8)
    print('\n \t using a distance of_:', d)
    for row in range(np.shape(img)[0]):
      for col in range(np.shape(img)[1]):
        #print('rrrrrrrooooooooooowwwwwwwwww........',row,'cooooooooooollllllll',col)
        slidingW = paddedImage[row:row+window,col:col+window]
        # plt.imshow(slidingW)
        # plt.show()
        io = mahotas.features.haralick(slidingW,distance=d,return_mean=True)
        #print(math.floor(window/2)+np.shape(img)[0]-1,row,col)
        HM[row,col,:]=io
        #print('\nSliding window:.............',np.shape(slidingW))
    print('Shape haralick cube',np.shape(HM))


  else:
    print('Making a int8 copy of the image')
    paddedImage[0+math.floor(window/2):np.shape(img)[0]+math.floor(window/2),0+math.floor(window/2):np.shape(img)[1]+math.floor(window/2)] = img
    paddedImage=paddedImage.astype(np.uint8)
    print('\n \t using a distance of:', d)
    for row in range(np.shape(img)[0]):
      for col in range(np.shape(img)[1]):
        #print('rrrrrrrooooooooooowwwwwwwwww........',row,'cooooooooooollllllll',col)
        slidingW = paddedImage[row:row+window,col:col+window]
        # plt.imshow(slidingW)
        # plt.show()
        io = mahotas.features.haralick(slidingW,distance=d,return_mean=True)
        #print(math.floor(window/2)+np.shape(img)[0]-1,row,col)
        HM[row,col,:]=io
        #print('\nSliding window:.............',np.shape(slidingW))
    print('Shape haralick cube',np.shape(HM))

  haralickFeatures=HM
  return haralickFeatures

