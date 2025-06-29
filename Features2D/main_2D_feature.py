# from feature_extract_invent import extract2DFeatureInfo
from featureStatistics import feature_stats
from scipy.stats import kurtosis, skew
from feature_extract_invent import extract2DFeatureInfo
import cv2 as cv
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
########## "ax_cor_sag": still on progress. Dont set
########## save_features: if True it saves all features and feature statistics (with feature names) in the outputh_path. If False it only show the results on terminal

def main2Dfeature(input_image,input_mask,output_path,ax_cor_sag = 'ax',windows=[3,5,7,11],save_features=True):

    if isinstance(input_image,str):
        print('\n \t Getting the image from dir:.....',input_image)
        image = cv.imread(input_image)
        mk = cv.imread(input_mask)

    else:
        print('\n \t Provided input is an image in a variable')
        image = input_image
        mk = input_mask

    all_features_names = []
    all_features_values =[]
    all_feature_stats = []
    all_features_names = pd.DataFrame(all_features_names)
    all_features_values = pd.DataFrame(all_features_values)
    all_feature_stats = pd.DataFrame(all_feature_stats)
    all_stats_names = []


    if len(np.shape(image))>2: #Check if volume 3D image or 2D
        if ax_cor_sag == 'ax':
            print('\n \t Processing a 3D image\n')

            for z in range(np.shape(image)[2]):
                ima = image[:,:,z]
                threshold = skimage.filters.threshold_otsu(mk[:,:,0])
                mask = mk[:,:,0] > threshold
                features = extract2DFeatureInfo(ima,mask,windows)
                # print('----------->',np.shape(features[2]),features[2])
                all_features_names = pd.DataFrame(features[0][0])
                all_features_values = pd.concat([all_features_values,pd.DataFrame(features[1][0])])
                all_feature_stats = pd.concat([all_feature_stats,pd.DataFrame(np.reshape(features[2][0],(1,len(features[2][0]))))])
                # print('_____________>',np.shape(all_feature_stats))

            print(np.shape(all_feature_stats))

    else:       # in case it is a volume or 2D image

         if ax_cor_sag == 'ax':
            print('\t Processing a 2D image')
            threshold = skimage.filters.threshold_otsu(mk)
            mask = mk > threshold
            features = extract2DFeatureInfo(image,mask,windows)
            # print('IMPORTANT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',features[0][0])
            all_features_names = pd.DataFrame(features[0][0])
            all_features_values = pd.DataFrame(features[1][0])
            all_feature_stats = pd.DataFrame(features[2][0])
            # print(features[3])


    # print(np.shape(features[3]))
    featureStatNames = pd.DataFrame(features[3])
    # all_features_names = np.unique(all_features_names)
    all_features_names = pd.DataFrame(all_features_names)
    if save_features == True:
        all_features_names.to_csv(output_path+'/2D_feature_names.csv')
        all_features_values.to_csv(output_path+'/2D_feature_values.csv')
        all_feature_stats.to_csv(output_path+'2D_feature_stats.csv')
        featureStatNames.to_csv(output_path+'/2D_feature_stats_names.csv')
    else:
        return all_features_names,all_features_values,all_feature_stats,featureStatNames

