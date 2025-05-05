from feature_extract_invent import extract2DFeatureInfo
from featureStatistics import feature_stats
from scipy.stats import kurtosis, skew
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

def main2Dfeature(input_image_path,input_mask_path,output_path,ax_cor_sag = 'ax',windows=[3,5,7,11],save_features=True):
    image = cv.imread(input_image_path)
    mk = cv.imread(input_mask_path)
    all_features_names = []
    all_features_values =[]
    all_features_names = pd.DataFrame(all_features_names)
    all_features_values = pd.DataFrame(all_features_values)
    if len(np.shape(image))>2:
        if ax_cor_sag == 'ax':
            for z in range(np.shape(image)[2]):
                ima = image[:,:,z]
                threshold = skimage.filters.threshold_otsu(mk[:,:,0])
                mask = mk[:,:,0] > threshold
                features = extract2DFeatureInfo(ima,mask,windows)
                print(features[1][0])
                all_features_names = pd.DataFrame(features[0][0])
                all_features_values = pd.concat([all_features_values,pd.DataFrame(features[1][0])])
            stats_features = feature_stats(features[1][0])
            stats_features = pd.DataFrame(stats_features)
    else:
         if ax_cor_sag == 'ax':
            threshold = skimage.filters.threshold_otsu(mk[:,:,0])
            mask = mk[:,:,0] > threshold
            features = extract2DFeatureInfo(image,mask,windows)
            print(features[1][0])
            all_features_names = pd.DataFrame(features[0][0])
            all_features_values = pd.DataFrame(features[1][0])
            stats_features = feature_stats(features[1][0])
            stats_features = pd.DataFrame(stats_features)



    all_features_names = np.unique(all_features_names)
    all_features_names = pd.DataFrame(all_features_names)
    if save_features == True:
        all_features_names.to_csv(output_path+'/2D_feature_names.csv')
        all_features_values.to_csv(output_path+'/2D_feature_values.csv')
        stats_features.to_csv(output_path+'/2D_feature_stats.csv')
    else:
        return all_features_names,all_features_values,stats_features

