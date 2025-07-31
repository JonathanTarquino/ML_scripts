# -*- coding: utf-8 -*-
"""feature_extract_invent.ipynb
"""
############################  REQUIREMENTS FOR THIS CODE
# pip install pyradiomics
# pip install mahotas
# pip install pip pyfeats
# pip install medviz

#from radiomics import featureextractor
from pathlib import Path
from scipy.ndimage import convolve1d,standard_deviation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
import math
import glob
import time
import skimage
from scipy.signal import convolve2d, medfilt2d
from scipy.ndimage.filters import generic_filter, median_filter, convolve1d, convolve
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

from medviz.feats import collage
from medviz.feats.collage.main import Collage, HaralickFeature
print(Collage)

####################################################################################################################

def boundingbox3(vol, mask=[], n=[],slice_opt=None,disp_opt=None):
  min_col = 100000000
  min_row = 100000000
  min_plane = 100000000
  max_col = 0
  max_row = 0
  max_plane = 0

  # make all zero values in the original volume nonzero

  volcopy = vol

  minval = np.max(vol.ravel())/10000
  vol[vol == 0] = minval

  if len(mask)>1: #use the mask to create a bounding box:
    if len(np.shape(mask))>=3:
      print('Provided binary mask...cropping volume')
      label_img = label(mask)
      regions = skimage.measure.regionprops(label_img)
      for props in regions:
        minr, minc, minp, maxr, maxc, maxp = props.bbox
        # print(minr, minc, minp, maxr, maxc, maxp)
        if min_col>minc:
          min_col=minc

        if min_row>minr:
          min_row=minr

        if min_plane>minp:
          min_plane = minp

        if max_col<maxc:
          max_col=maxc

        if max_row<maxr:
          max_row=maxr

        if max_plane < maxp:
          max_plane = maxp

      croppedV = volcopy[min_row-n:max_row+n,min_col-n:max_col+n, min_plane-n: max_plane+n ]
      croppedM = mask[min_row-n:max_row+n,min_col-n:max_col+n, min_plane-n:max_plane+n]

  return croppedV,croppedM

###############################################################################################################

def extract3DFeatIntensities(featVol, mask, vals=1):

  # %INPUTS
  # % featVol = 3-D volume of texture features
  # % mask = 2-D annotation image
  # % vals = vector of mask values to find intensities within

  # % OUTPUT
  # % varargout = array(s) of intensities. Number based on length of vals

  joinFeat = []
  joinFeat = pd.DataFrame(joinFeat)

  if (not featVol.all) or (not mask.all) or (not vals):
        return []

  print('Extracting single feature matrix:',sum(mask.ravel()),np.shape(featVol))
  if len(featVol.shape) >2:
    f = featVol #single feature volume
    for j in range(vals):# loop through all mask values
      joinFeat = f[mask==vals].ravel(order='F')
  #   print(f,'\n',joinFeat)
      print(np.shape(joinFeat))
    return joinFeat
  else:
    for j in range(vals):
      for col in range(featVol.shape[1]):
        f = featVol[:,col]
        joinFeat = pd.concat([joinFeat,pd.DataFrame(f[mask.flatten()==vals])],axis=1)
    return joinFeat

######################################################################################################################################

def rangefilt(I,window):
  if I.dtype in ['uint8','uint16']:
    kernel= footprint_rectangle((window,window,window))
    imgOutput = gradient(I,kernel)
    #print('Rangefilt output',imgOutput)
  else:
    Im=I.astype(np.uint8)
    kernel= footprint_rectangle((window,window,window))
    imgOutput = gradient(Im,kernel)
    #print('Rangefilt output',imgOutput)

  return imgOutput

#####################################################################################################################################
def convolve3d(vol,knl):

  # print('\n---------->',knl.shape)

  # Convolve over all three axes in a for loop
  out =vol.copy()

  out1 = convolve(vol,knl)

  return out1



######################################################################################################################################
def grayfilts3(img,WindowSize=3):
  print(':::::::::::::::::::::::::')
  # grayFilt = []
  # grayFilt = pd.DataFrame(grayFilt)
  imShape = np.shape(img)

  if len(imShape)>3:
    print('\n Only 3D images supported and 4D provided \n')
    return ()
  else:
    grayFilt = np.zeros((np.shape(img)[0]*np.shape(img)[1]*np.shape(img)[2],4))

    # Calculating Mean image
    print('\nCalculating Mean Image.\n')
    kernel = np.ones([WindowSize,WindowSize,WindowSize])/(WindowSize*WindowSize*WindowSize)
    filtResult = convolve3d(img, kernel)
    grayFilt[:,0] = filtResult.flatten()
    # grayFilt = pd.concat([grayFilt,pd.DataFrame(filtResult)])

    # Calculating Median image
    print('\nCalculating Median Image.\n')
    filtResult = median_filter(img,[WindowSize,WindowSize,WindowSize])
    grayFilt[:,1] = filtResult.flatten()
    # grayFilt = pd.concat([grayFilt,pd.DataFrame(filtResult)])

    # Calculating windowed standard deviation filter
    print('\nCalculating std Image.\n')
    filtResult = generic_filter(img, np.std, [WindowSize,WindowSize,WindowSize])
    grayFilt[:,2] = filtResult.flatten()
    # grayFilt = pd.concat([grayFilt,pd.DataFrame(filtResult)])

    # Calculating windowed range filter
    print('\n Calculating Windowed range image.\n')
    filtResult = rangefilt(img, WindowSize)
    grayFilt[:,3] = filtResult.flatten()
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

def sobelxydiag3(Vol):
  sobeli1 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobeli2 = [[0, 3, 6], [-3, 0, 3], [-6, -3, 0]]
  sobeli3 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobel= np.dstack((sobeli1,sobeli2,sobeli3))
  Y = ndi.convolve(Vol,sobel)
  # print(Vol.shape,Y.shape,sobel.shape)
  return Y

def sobelyxdiag3(Vol):
  sobeli1 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobeli2 = [[0, 3, 6], [-3, 0, 3], [-6, -3, 0]]
  sobeli3 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobel= np.dstack((sobeli1,sobeli2,sobeli3))
  sobel= np.flip(sobel,0)
  Y = ndi.convolve(Vol,sobel)
  # print(Vol.shape,Y.shape,sobel.shape)
  return Y


def sobelyzdiag3(Vol):
  sobeli1 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobeli2 = [[0, 3, 6], [-3, 0, 3], [-6, -3, 0]]
  sobeli3 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobel= np.dstack((sobeli1,sobeli2,sobeli3))
  # print('sobel:',sobel)
  sobel= np.moveaxis(sobel,source=[0,1,2],destination=[0,2,1])
  # print(sobel)
  Y = ndi.convolve(Vol,sobel)
  # print(Vol.shape,Y.shape,sobel.shape)
  return Y

def sobelzydiag3(Vol):
  sobeli1 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobeli2 = [[0, 3, 6], [-3, 0, 3], [-6, -3, 0]]
  sobeli3 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobel= np.dstack((sobeli1,sobeli2,sobeli3))
  sobel= np.moveaxis(sobel,[0,1,2],[0,2,1])
  sobel=np.flip(sobel,0)
  Y = ndi.convolve(Vol,sobel)
  # print(Vol.shape,Y.shape,sobel.shape)
  return Y
#
def sobelxzdiag3(Vol):
  sobeli1 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobeli2 = [[0, 3, 6], [-3, 0, 3], [-6, -3, 0]]
  sobeli3 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobel= np.dstack((sobeli1,sobeli2,sobeli3))
  sobel= np.moveaxis(sobel,[0,1,2],[2,1, 0])
  Y = ndi.convolve(Vol,sobel)
  # print(Vol.shape,Y.shape,sobel.shape)
  return Y
#
def sobelzxdiag3(Vol):
  sobeli1 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobeli2 = [[0, 3, 6], [-3, 0, 3], [-6, -3, 0]]
  sobeli3 = [[0, 1, 3], [-1, 0, 1], [-3, -1, 0]]
  sobel= np.dstack((sobeli1,sobeli2,sobeli3))
  sobel= np.moveaxis(sobel,[0,1,2],[2,1, 0])
  sobel= np.flip(sobel,0)
  Y = ndi.convolve(Vol,sobel)
  # print(Vol.shape,Y.shape,sobel.shape)
  return Y





###########################

def gradfilts3(img):
  imSize = np.shape(img)
  print('Image dimensions:',len(imSize))
  if len(imSize)>3:
    print('Error: only 3D images are supported, provided:', imSize)
    return []

  else:
    # feat_names = ['Gradient sobelx','Gradient sobely','Gradient sobelxy','Gradient sobelyx','Gradient x','Gradient y','Gradient magnitude','Gradient dx','Gradient dy','Gradient diagonal']
    feat_names = ['Gradient sobelx','Gradient sobely','Gradient sobelz',
                  'Gradient sobelxy','Gradient sobelyx','Gradient sobelxz',
                  'Gradient sobelzx','Gradient sobelyz','Gradient sobelzy',
                  'Gradient x','Gradient y','Gradient z','Gradient magnitude']

    nfeatures=len(feat_names);
    gradfeats=np.ones([imSize[0]*imSize[1]*imSize[2], nfeatures]);


    print('Calculating x,y,z Sobel edge images.\n')
    gradfeats[:,0] = np.reshape(ndi.sobel(img, axis=0),(imSize[0]*imSize[1]*imSize[2],))
    gradfeats[:,1] = np.reshape(ndi.sobel(img, axis=1),(imSize[0]*imSize[1]*imSize[2],))
    gradfeats[:,2] = np.reshape(ndi.sobel(img, axis=2),(imSize[0]*imSize[1]*imSize[2],))


    print('Calculating diagonal Sobel edge images.\n')
    gradfeats[:,3] = np.reshape(sobelxydiag3(img),(imSize[0]*imSize[1]*imSize[2],))
    gradfeats[:,4] = np.reshape(sobelyxdiag3(img),(imSize[0]*imSize[1]*imSize[2],))
    gradfeats[:,5] = np.reshape(sobelyzdiag3(img),(imSize[0]*imSize[1]*imSize[2],))
    gradfeats[:,6] = np.reshape(sobelzydiag3(img),(imSize[0]*imSize[1]*imSize[2],))
    gradfeats[:,7] = np.reshape(sobelxzdiag3(img),(imSize[0]*imSize[1]*imSize[2],))
    gradfeats[:,8] = np.reshape(sobelzxdiag3(img),(imSize[0]*imSize[1]*imSize[2],))

    print('Calculating directional and magnitude gradients.\n')
    # ,gradfeats[:,10],gradfeats[:,11] = np.gradient(img)
    sd = np.gradient(img)

    gradfeats[:,9]= np.reshape(sd[0],(imSize[0]*imSize[1]*imSize[2],))
    gradfeats[:,10]= np.reshape(sd[1],(imSize[0]*imSize[1]*imSize[2],))
    gradfeats[:,11]= np.reshape(sd[2],(imSize[0]*imSize[1]*imSize[2],))

    gradfeats[:,12] = np.sqrt(gradfeats[:,9]**2 +gradfeats[:,10]**2+gradfeats[:,11]**2)

    return gradfeats

############################# Haralick for 3D images #################################################################################

def extractHaralick(img,window,d=1):

  paddedImage = np.zeros((np.shape(img)[0]+(2*window),np.shape(img)[1]+(2*window),np.shape(img)[2]+(2*window)))
  print('Padded image shape:',np.shape(img),' to ',np.shape(paddedImage))
  HM = np.zeros((np.shape(img)[0]*np.shape(img)[1]*np.shape(img)[2],13))

  if img.dtype in ['uint8','uint16']:
    print('\n Making a int8 copy of the image')
    paddedImage[0+math.floor(window/2):np.shape(img)[0]+math.floor(window/2),
                0+math.floor(window/2):np.shape(img)[1]+math.floor(window/2),
                0+math.floor(window/2):np.shape(img)[2]+math.floor(window/2)] = img
    paddedImage=paddedImage.astype(np.uint8)
    print('\n \t using a distance of_:', d)
    for row in range(np.shape(img)[0]):
      for col in range(np.shape(img)[1]):
        #print('rrrrrrrooooooooooowwwwwwwwww........',row,'cooooooooooollllllll',col)
        slidingW = paddedImage[row:row+window,col:col+window]
        # plt.imshow(slidingW)
        # plt.show()
        io = mahotas.features.haralick(slidingW,distance=d,return_mean=True)
        print(io.shape)
        #print(math.floor(window/2)+np.shape(img)[0]-1,row,col)
        HM[row,col,:]=io
        #print('\nSliding window:.............',np.shape(slidingW))
    print('Shape haralick cube',np.shape(HM))


  else:
    print('Making a int8 copy of the image')
    paddedImage[math.floor(window/2):np.shape(img)[0]+math.floor(window/2),
                math.floor(window/2):np.shape(img)[1]+math.floor(window/2),
                math.floor(window/2):np.shape(img)[2]+math.floor(window/2)] = img
    paddedImage=paddedImage.astype(np.uint8)
    print('\n \t using a distance of:', d)
    for row in range(np.shape(img)[0]):
      for col in range(np.shape(img)[1]):
        for pl in range(np.shape(img)[2]):
          #print('rrrrrrrooooooooooowwwwwwwwww........',row,'cooooooooooollllllll',col)
          slidingW = paddedImage[row:row+window,col:col+window,pl:pl+window]
          # plt.imshow(slidingW)
          # plt.show()
          io = mahotas.features.haralick(slidingW,distance=d,return_mean=True)  # Computing Haralick's features on thw sliding window'
          # print(io)
          #print(math.floor(window/2)+np.shape(img)[0]-1,row,col)
          HM[row*col*pl,:]=io
          #print('\nSliding window:.............',np.shape(slidingW))
    print('Shape haralick cube',np.shape(HM))

  haralickFeatures=HM
  return haralickFeatures


############################################ For Gabor 3D ###################################
import math
import numpy as np
# from mayavi import mlab

def rotation(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def gabor_fn(sigma, thetas, Lambda, psi, gamma, size, plot=False, slices=False):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    sigma_z = float(sigma) / gamma

    # Bounding box
    (z, y, x) = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1), np.arange(-size, size +1))

    # Rotation
    R = rotation(thetas)
    z_prime = z * R[0,0] + y * R[0,1] + x * R[0,2]
    y_prime = z * R[1,0] + y * R[1,1] + x * R[1,2]
    x_prime = z * R[2,0] + y * R[2,1] + x * R[2,2]

    gb = np.exp(-.5 * (x_prime ** 2 / sigma_x ** 2 + y_prime ** 2 / sigma_y ** 2 + z_prime ** 2 / sigma_z)) * np.cos(2 * np.pi * x_prime / Lambda + psi)

    # if plot:
    #
    #     if slices:
    #         mlab.volume_slice(gb, plane_orientation='x_axes', slice_index=31)
    #         mlab.volume_slice(gb, plane_orientation='y_axes', slice_index=31)
    #         mlab.volume_slice(gb, plane_orientation='z_axes', slice_index=31)
    #         mlab.show()
    #
    #     mlab.contour3d(gb)
    #     mlab.show()

    return gb

def gaussian_fn(sigma, size=31, plot=False):
    (z, y, x) = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1), np.arange(-size, size +1))

    g = np.exp(-(x**2/float(size)+y**2/float(size)+z**2/float(size)))
    gauss = g / g.sum()

    # if plot:
    #      mlab.volume_slice(g, plane_orientation='x_axes', slice_index=31)
    #      mlab.volume_slice(g, plane_orientation='y_axes', slice_index=31)
    #      mlab.volume_slice(g, plane_orientation='z_axes', slice_index=31)
    #      mlab.show()

    return gauss

def filter_bank_gb3d(sigma=4.0, Lambda=10.0, psi=0.3, gamma=0.3, size=31, plot=False):
    filters = []
    gabor_names = list(['Gabor_0'])
    sigma = sigma
    for theta_x in np.arange(0, np.pi, np.pi / 4):
       for theta_y in np.arange(0, np.pi, np.pi / 4):
           for theta_z in np.arange(0, np.pi, np.pi / 4):
               thetas = [theta_x, theta_y, theta_z]
               # print(thetas)
               kern = gabor_fn(sigma, thetas, Lambda, psi, gamma, size)
               kern /= 1.5*kern.sum()
               filters.append(np.transpose(kern))
               gabor_names.append('Gabor_x_'+str(theta_x)+'_y_'+str(theta_y)+'_z_'+str(theta_z)+'_sigma_Lambda_'+str(sigma)+'_'+str(Lambda))

    filters.append(gaussian_fn(sigma, size=size, plot=plot))

    return filters,gabor_names
##############################################3

def compute_gabor(image, kernels):
    print('Number of kernels:',len(kernels))
    feats = np.zeros((np.shape(image)[0]*np.shape(image)[1]*np.shape(image)[2],len(kernels)), dtype=np.double)
    for k, kernel in enumerate(kernels):
        # print(kernel.shape,k)
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[:,k] = np.reshape(filtered,(np.shape(image)[0]*np.shape(image)[1]*np.shape(image)[2],))
        # gabor_names.append(['GaborXY_theta= pi*'+str(theta)+'/8_lambda='+str(frequency)])
    return feats

def gaborFilter(img):
  # preparing filter bank kernels

  ima = img_as_float(img)
  kernels, gabor_names = filter_bank_gb3d(size=3)
  gabor_feats = compute_gabor(ima,kernels )
  # print('------------------------------------',np.shape(gabor_feats),gabor_names)
  return(gabor_feats,gabor_names)

#############################################################################################################



def compute_collage3d(
    image: np.ndarray,
    mask: np.ndarray,
    haralick_windows: int,
    feature_maps=False,

) -> np.ndarray:
    feats = {}
    descriptors = [
        "Collage_AngularSecondMoment",  # 0
        "Collage_Contrast",  # 1
        "Collage_Correlation",  # 2
        "Collage_SumOfSquareVariance",  # 3
        "Collage_SumAverage",  # 4
        "Collage_SumVariance",  # 5
        "Collage_SumEntropy",  # 6
        "Collage_Entropy",  # 7
        "Collage_DifferenceVariance",  # 8
        "Collage_DifferenceEntropy",  # 9
        "Collage_InformationMeasureOfCorrelation1",  # 10
        "Collage_InformationMeasureOfCorrelation2",  # 11
        "Collage_MaximalCorrelationCoefficient",  # 12
    ]
    print(image.shape)
    try:
        collage = Collage(
            image,
            mask,
            svd_radius=5,
            verbose_logging=True,
            num_unique_angles=64,
            haralick_window_size=haralick_windows,
        )
        print('Executing collage...')
        collage_feats = collage.execute()

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',collage_feats.shape)
        new_collage = np.zeros((collage_feats.shape[0]*collage_feats.shape[1]*collage_feats.shape[2],collage_feats.shape[3]*collage_feats.shape[4]),dtype= float)
        print(new_collage.shape)
        colu = 0
        for orientat in range(collage_feats.shape[4]):
          for ft in range (collage_feats.shape[3]):
            # print(colu,np.shape(collage_feats[:,:,ft,orientat]),collage_feats.shape[0]*collage_feats.shape[1]*collage_feats.shape[2])
            new_collage[:,colu] = np.reshape(collage_feats[:,:,:,ft,orientat], (collage_feats.shape[0]*collage_feats.shape[1]*collage_feats.shape[2],))
            colu +=1

        print(new_collage)

        # if save_raw_path:
        #     np.save(save_raw_path, collage_feats)

        # if feature_maps:
        #     which_features = [
        #         HaralickFeature.AngularSecondMoment,
        #         HaralickFeature.Contrast,
        #         HaralickFeature.Correlation,
        #         HaralickFeature.SumOfSquareVariance,
        #         HaralickFeature.SumAverage,
        #         HaralickFeature.SumVariance,
        #         HaralickFeature.SumEntropy,
        #         HaralickFeature.Entropy,
        #         HaralickFeature.DifferenceVariance,
        #         HaralickFeature.DifferenceEntropy,
        #         HaralickFeature.InformationMeasureOfCorrelation1,
        #         HaralickFeature.InformationMeasureOfCorrelation2,
        #         HaralickFeature.MaximalCorrelationCoefficient,
        #     ]

        #     alpha = 0.5
        #     extent = 0, image.shape[1], 0, image.shape[0]

        #     for which_feature in which_features:
        #         collage_output = collage.get_single_feature_output(which_feature)

        #         figure = plt.figure(figsize=(15, 15))
        #         plt.imshow(image, cmap=plt.cm.gray, extent=extent)
        #         plt.imshow(collage_output, cmap=plt.cm.jet, alpha=alpha, extent=extent)

        #         figure.axes[0].get_xaxis().set_visible(False)
        #         figure.axes[0].get_yaxis().set_visible(False)

        #         plt.title(f"Feature map: {which_feature.name}")

        #         save_path_name = f"{save_path_feature_maps}_{which_feature.name}.png"
        #         plt.savefig(save_path_name)
        #         plt.close()

        for collage_idx, descriptor in enumerate(descriptors):
            #print(f"Processing collage {descriptor}")
            feat = collage_feats[:, :, collage_idx].flatten()
            feat = feat[~np.isnan(feat)]

            feats[f"col_des_{descriptor}"] = [
                feat.mean(),
                feat.std(),
                skew(feat),
                kurtosis(feat),
            ]

    except ValueError as err:
        print(f"VALUE ERROR- {err}")

    except Exception as err:
        print(f"EXCEPTION- {err}")

    return feats,new_collage,descriptors




#################################################################################################################################

def _image_xor(f):
    # Turn "0" to "1" and vice versa: XOR with image consisting of "1"s
    f = f.astype(np.uint8)
    mask = np.ones(f.shape, np.uint8)
    out = np.zeros(f.shape, np.uint8)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
          for z in range(f.shape[2]):
            out[i,j,z] = f[i,j,z] ^ mask[i,j,z]
    return out


##############################################################
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

    if mask is None:
        mask = np.ones(f.shape)
        print('----------',np.shape(mask))

    # 1) Labels
    labels = ["LTE_LL","LTE_LE","LTE_LS","LTE_EL","LTE_EE","LTE_ES","LTE_SL","LTE_SE","LTE_SS"]
    labels = [label+'_w_'+str(l) for label in labels]

    # 2) Parameters
    f = np.array(f, np.double)
    mask = np.array(mask, np.double)
    kernels = np.zeros((l,l,l,9), np.double) # canvas for filter

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

    oneskernel = np.ones((l,l,l), np.double)

    kernels[:,:,:,0] = np.multiply(L.reshape(-1,1),L) # LL kernel
    kernels[:,:,:,1] = np.multiply(L.reshape(-1,1),E) # LE kernel
    kernels[:,:,:,2] = np.multiply(L.reshape(-1,1),S) # LS kernel
    kernels[:,:,:,3] = np.multiply(E.reshape(-1,1),L) # EL kernel
    kernels[:,:,:,4] = np.multiply(E.reshape(-1,1),E) # EE kernel
    kernels[:,:,:,5] = np.multiply(E.reshape(-1,1),S) # ES kernel
    kernels[:,:,:,6] = np.multiply(S.reshape(-1,1),L) # SL kernel
    kernels[:,:,:,7] = np.multiply(S.reshape(-1,1),E) # SE kernel
    kernels[:,:,:,8] = np.multiply(S.reshape(-1,1),S) # SS kernel

    # 4) Get mask where convolution should be performed
    mask_c = _image_xor(mask)
    mask_conv = convolve3d(mask_c, oneskernel)
    mask_conv = np.abs(np.sign(mask_conv)-1)
    print(np.shape(mask_conv))

    # 5) Calculate energy of each convolved image with each kernel: total 9
    energy = np.zeros((np.shape(mask_conv)[0]*np.shape(mask_conv)[1]*np.shape(mask_conv)[2],9),np.double)
    area = sum(sum(mask_conv))
    for i in range(9): # 9 is the nuber of kernels
        f_conv = convolve3d(f, kernels[:,:,:,i])
        f_conv = np.multiply(f_conv,mask_conv)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',np.shape(f_conv))
        energy[:,i] = f_conv.flatten()
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

    return energy, labels

##############################################################################################

def extract3DFeatureInfo(img_org,mask_org,ws_options=[3,5,7,9,11],class_options = ['raw','gray','gradient','haralick','gabor','laws','collage'], with_stats=True):
  # im_org =        numpy array (3D or 2D), containing the original image array
  # mask_org =      numpy array (3D or 2D), containing the original mask array
  # class_options = (OPTIONAL) array of strings corresponding to desired feature classes:
  #                 DEFAULT: class_options = ['raw','gray','gradient','haralick','gabor','laws','collage'] a list like variable containing feature familly names to be extracted
  # ws_options = (  OPTIONAL) array of integers corresponding to desired window levels:
  #                 DEFAULT: ws_options = [3, 5, 7, 9, 11]
  # with_stats =    ( OPTIONAL) logic, if True it set the function to provide feature statistics togheter with the pixel wise feature information
  #                 DEFAULT: tRUE



  #Initialization
  statistics = []
  matrixNames = []
  matrixNames = pd.DataFrame(matrixNames)
  matrixFeatures = []
  matrixFeatures = pd.DataFrame(matrixFeatures)
  statFeatures = []
  statFeatures = pd.DataFrame(statFeatures)
  statFeatureNames =[]
  statFeatures = pd.DataFrame(statFeatureNames)

  ##RECOMMENDED: CROP IMAGE AND MASK (saves time and memory)!
  [img,mask] = boundingbox3(img_org,mask_org,np.max(ws_options))
  # plt.imshow(mask[0,::])
  # plt.show()

  print('\n Cropped Image shape:',np.shape(img))

  ##2D Feature Intensity Extraction
  print('\n Extracting Features for provide image and mask volume')

  #------------- Raw intensity features ----------------

  if 'raw' in class_options:
    print('\n \t Extracting Raw Features ...\n')
    feat_vect = extract3DFeatIntensities(img, mask)
    print(feat_vect)

    if with_stats == True:
      statFeatures = pd.concat([statFeatures,pd.DataFrame([feat_vect.mean(),feat_vect.std(),skew(feat_vect),kurtosis(feat_vect)])],axis = 1)
      statistics = ['Mean_of_','std_of_','skew_of_','kurtosis_of_']



    matrixFeatures = pd.concat([pd.DataFrame(feat_vect),matrixFeatures], axis = 0)
    matrixNames = pd.concat([pd.DataFrame(['Raw intensity']),matrixNames])



  #--------------Gray Level Statistics----------------%
  if 'gray' in class_options:
    print('\n Extracting Gray Level Statistics:\n')
    grayFeats = []
    grayFeats = pd.DataFrame(grayFeats)

    for ws in ws_options:
      print('\n \t Using a window size of ',ws,'\n')
      gf = grayfilts3(img,ws)
      feat_names = ['Mean Image_','Median Image_','std Image_','Windowed range image_']
      feat_vect = extract3DFeatIntensities(gf, mask)
      print('Gray level Feature vector statistics:...............',np.shape(feat_vect))
      print('\n featvect\n',feat_vect)
      f_names = [x+'_w'+str(ws) for x in feat_names]
      matrixFeatures = pd.concat([matrixFeatures,pd.DataFrame(feat_vect)], axis=1)
      matrixNames = pd.concat([matrixNames,pd.DataFrame(f_names)],axis = 0)

      if with_stats == True:
          for f_idx in range(feat_vect.shape[1]):
            statFeatures = pd.concat([statFeatures,pd.DataFrame([feat_vect.iloc[:,f_idx].mean(),
                                                                feat_vect.iloc[:,f_idx].std(),
                                                                skew(feat_vect.iloc[:,f_idx]),
                                                                kurtosis(feat_vect.iloc[:,f_idx])])], axis = 0)


      print('xxxxxxxxxxxxxxxx',np.shape(matrixFeatures),np.shape(matrixNames),np.shape(statFeatures))
      print(matrixNames,'\n',statFeatureNames)

    # ------------------ Haralick features ------------------------------
  if 'haralick' in class_options:

    start = time.time()

    print('\nExtracting Haralick-based (GLCM) features:\n')
    haralick_labels = ["Haralick Angular Second Moment",
                  "Haralick Contrast",
                  "Haralick Correlation",
                  "Haralick Sum of Squares: Variance",
                  "Haralick Inverse Difference Moment",
                  "Haralick Sum Average",
                  "Haralick Sum Variance",
                  "Haralick Sum Entropy",
                  "Haralick Entropy",
                  "Haralick Difference Variance",
                  "Haralick Difference Entropy",
                  "Haralick Information Measure of Correlation 1",
                  "Haralick Information Measure of Correlation 2"]
    HaralickFeat = extractHaralick(img,ws)

    end = time.time()
    print('Elapsed time for Haralick:',end - start)
    feat_vect = extract3DFeatIntensities(HaralickFeat, mask)
    #print('Haralick feature vector statistics:.................\n',np.shape(feat_vect))
    f_names = [x+'_w'+str(ws) for x in haralick_labels]
    matrixFeatures = pd.concat([matrixFeatures,pd.DataFrame(feat_vect)], axis=1)
    matrixNames = pd.concat([matrixNames,pd.DataFrame(f_names)],axis = 0)
    if with_stats == True:
            for f_idx in range(feat_vect.shape[1]):
              statFeatures = pd.concat([statFeatures,pd.DataFrame([feat_vect.iloc[:,f_idx].mean(),
                                                                  feat_vect.iloc[:,f_idx].std(),
                                                                  skew(feat_vect.iloc[:,f_idx]),
                                                                  kurtosis(feat_vect.iloc[:,f_idx])])])

    print('xxxxxxxxxxxxxxxx',np.shape(matrixFeatures),np.shape(statFeatures),'\n',matrixNames)





  #---------------- Laws--------------------------------------
  if 'laws' in class_options:
    print('\nExtracting Laws features:\n')
    Laws_out = lte_measures(img_org, l=ws)
    print('+++++++++++++++++++++++++++++++++++++++++++\n',np.shape(Laws_out[0]))
    feat_vect = extract3DFeatIntensities(Laws_out[0], mask_org)
    f_names = Laws_out[1]
    print('\n featvect\n',feat_vect)
    matrixFeatures = pd.concat([matrixFeatures,pd.DataFrame(feat_vect)], axis=1)
    matrixNames = pd.concat([matrixNames,pd.DataFrame(f_names)],axis = 0)
    print(matrixFeatures)

    if with_stats == True:
            for f_idx in range(feat_vect.shape[1]):
              statFeatures = pd.concat([statFeatures,pd.DataFrame([feat_vect.iloc[:,f_idx].mean(),
                                                                  feat_vect.iloc[:,f_idx].std(),
                                                                  skew(feat_vect.iloc[:,f_idx]),
                                                                  kurtosis(feat_vect.iloc[:,f_idx])])])

    print('xxxxxxxxxxxxxxxx',np.shape(matrixFeatures),np.shape(matrixNames),np.shape(statFeatures))
    print(matrixNames,'\n',statFeatureNames)

  # ---------------- Collage ------------------------------------------
  if 'collage' in class_options:

    start = time.time()
    print('\nExtracting Collage features:\n', img_org.shape)
    # due to limitation in the minumum patch size of collage pack use "img_org" instead of the cropped version "img"
    mask_complete = np.ones(np.shape(img_org)) # a fake mask is used to evaluate the whole image
    new_im = np.zeros((np.shape(img)[0]+60,np.shape(img)[1]+60,np.shape(img)[2]+60))
    new_mask = np.zeros((np.shape(img)[0]+60,np.shape(img)[1]+60,np.shape(img)[2]+60))
    new_im[30:np.shape(img)[0]+30,30:np.shape(img)[1]+30,30:np.shape(img)[2]+30] = img
    new_mask[30:np.shape(img)[0]+30,30:np.shape(img)[1]+30,30:np.shape(img)[2]+30] = mask

    coll = compute_collage3d(new_im,new_mask, haralick_windows=ws)
    end = time.time()
    print('Elapsed time for Collage:',end-start , np.shape(coll[1]))

    feat_vect = extract3DFeatIntensities(coll[1], new_mask)
    print(coll[2],'\n featvect\n',feat_vect)
    f_names = [x+'_w'+str(ws) for x in coll[2]]
    matrixFeatures = pd.concat([matrixFeatures,pd.DataFrame(feat_vect)], axis=1)
    matrixNames = pd.concat([matrixNames,pd.DataFrame(f_names)],axis = 0)


    if with_stats == True:
      for f_idx in range(feat_vect.shape[1]):
        statFeatures = pd.concat([statFeatures,pd.DataFrame([feat_vect.iloc[:,f_idx].mean(),
                                                            feat_vect.iloc[:,f_idx].std(),
                                                            skew(feat_vect.iloc[:,f_idx]),
                                                            kurtosis(feat_vect.iloc[:,f_idx])])], axis = 0)
    print('xxxxxxxxxxxxxxxx',np.shape(matrixFeatures),'\n',matrixFeatures,'\n',statFeatures.shape,matrixNames)
    print(matrixNames,'\n',statFeatureNames)



  #------------- Gradient ------------------------------
  if 'gradient' in class_options:
    print('\nExtracting Gradient features:\n')
    feat_names = ['Gradient sobelx','Gradient sobely','Gradient sobelxy','Gradient sobelyx','Gradient x','Gradient y','Gradient magnitude','Gradient dx','Gradient dy','Gradient diagonal']
    gradOut = gradfilts3(img)
    # print('Grad results:\n',gradOut)
    f_names = [x+'_w'+str(ws) for x in feat_names]
    feat_vect = extract3DFeatIntensities(gradOut, mask)
    # print('Gray level statistics:\n',np.shape(feat_vect),feat_vect[0:100])
    matrixFeatures = pd.concat([matrixFeatures,pd.DataFrame(feat_vect)], axis=1)
    matrixNames = pd.concat([matrixNames,pd.DataFrame(f_names)],axis = 0)

    if with_stats == True:
      for f_idx in range(feat_vect.shape[1]):
        statFeatures = pd.concat([statFeatures,pd.DataFrame([feat_vect.iloc[:,f_idx].mean(),
                                                            feat_vect.iloc[:,f_idx].std(),
                                                            skew(feat_vect.iloc[:,f_idx]),
                                                            kurtosis(feat_vect.iloc[:,f_idx])])], axis = 0)
    print('xxxxxxxxxxxxxxxx',np.shape(matrixFeatures),'\n',matrixFeatures,'\n',statFeatures.shape,matrixNames)
    print(matrixNames,'\n',statFeatureNames)




  #---------------- Gabor -------------------------------------------

  if 'gabor' in class_options:
    print('\n \t Extracting Gabor features\n')
    gabor_matrix = gaborFilter(img)
    feat_vect = extract3DFeatIntensities(gabor_matrix[0], mask)
    matrixFeatures = pd.concat([matrixFeatures,pd.DataFrame(feat_vect)], axis=1)
    matrixNames = pd.concat([matrixNames,pd.DataFrame(gabor_matrix[1])],axis = 0)


    if with_stats == True:
      for f_idx in range(feat_vect.shape[1]):
        statFeatures = pd.concat([statFeatures,pd.DataFrame([feat_vect.iloc[:,f_idx].mean(),
                                                            feat_vect.iloc[:,f_idx].std(),
                                                            skew(feat_vect.iloc[:,f_idx]),
                                                            kurtosis(feat_vect.iloc[:,f_idx])])], axis = 0)
    print('xxxxxxxxxxxxxxxx',np.shape(matrixFeatures),'\n',matrixFeatures,'\n',statFeatures.shape,matrixNames)
    print(matrixNames,'\n',statFeatureNames)

# ------------------ prost proc to return features-------------------------------------------

    for name in range(len(matrixNames.iloc[:,0])):
          # print('--------->',matrixNames.iloc[name,0])
          # print([r+matrixNames.iloc[name,0] for r in statistics])
          statFeatureNames = pd.concat([pd.DataFrame([r+matrixNames.iloc[name,0] for r in statistics]),pd.DataFrame(statFeatureNames)])

    return (matrixNames,matrixFeatures,statFeatures,statFeatureNames)
