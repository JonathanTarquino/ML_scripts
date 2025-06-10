from feature_extract_invent3D import extract3DFeatureInfo
from featureStatistics import feature_stats
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.stats import kurtosis, skew
from itk2numpy_tools import sitk2numpy_xyz

import skimage
import numpy as np
import pandas as pd
import os


##################################################################################################
####  #########  To use this function from main_2D_feature import main2Dfeature #################
#################################################################################################
## Parameters
## Inputs:
########## "input_image_path": corresponds to the actual path where the image is stored (image needs to be a numpy structure)
########## "input_mask_path": correspondsto the catual path where the mask is stored (needs to be binary numpy structure array)
########## "output_path": is the path where the user is expecting to save the features, in case save_features == True.
###########################Other way, it is required to run but no result is going to be saved there, and the output is returned as a variable
########## "windows": list of window sizes to apply windowed analysis

def main3Dfeature(input_image_path,input_mask_path,output_path,ax_cor_sag = 'ax',windows=[3,5,7,11],save_features=True):
    if '.mha' in input_image_path:
        image_type = '.mha'
        print('\n Image type:...........................',image_type)
        image_type = '.mha'

        # Reding Volume Image
        inputImage = sitk.ReadImage(input_image_path)
        spaceIm = inputImage.GetSpacing()
        orgIm = inputImage.GetOrigin()
        directIm = inputImage.GetDirection()
        volumeIm = sitk.GetArrayFromImage(inputImage)
        volumeIm = sitk2numpy_xyz(volumeIm) # This line is crucial since itk uses xyz format while numpy uses zyx for non-RGB images
        print('Image size:',volumeIm.shape,type(volumeIm))

        # Reading Label volume
        maskVol = sitk.ReadImage(input_mask_path)
        maskVol_array = sitk.GetArrayFromImage(maskVol)
        maskVol_array = sitk2numpy_xyz(maskVol_array)
        print('Mask size:',maskVol_array.shape,volumeIm.shape)

        # Check if teh shape of both, image volume and label volume, agree!
        if volumeIm.shape != maskVol_array.shape:
            print('Error: volume image and provided label mask has different shape')
            return []

        mask_shape = np.shape(maskVol_array)
        maskVol_array = np.uint8(maskVol_array/np.max(maskVol_array.ravel()))
        maskVol_array = np.reshape(maskVol_array,mask_shape)

        plt.imshow(volumeIm[:,:,0])
        plt.imshow(maskVol_array[:,:,0],alpha=0.4)
        plt.show()

        # Setting variables for features
        all_features_names = []
        all_features_values =[]
        all_features_names = pd.DataFrame(all_features_names)
        all_features_values = pd.DataFrame(all_features_values)


        features = extract3DFeatureInfo(volumeIm,maskVol_array,windows)


    elif '.nii' in path:
        image_type = '.nii'



    else:
        image_type = 'Other'




