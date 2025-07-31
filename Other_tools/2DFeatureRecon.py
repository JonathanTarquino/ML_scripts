import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/home/jonathan/Documents/Invent/Code_traslation/Classification_scr/')
sys.path.append('/home/jonathan/Documents/Invent/Code_traslation/feature_extraction_ML/Features2D/')
from corr_prunning import pick_best_uncorrelated_features
from CrossValidation import nFoldCV_withFS
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from CrossValidation import main_classifier
from performance import performance_values
from main_2D_feature import main2Dfeature
import seaborn as sns



def sitk2numpy_xyz(volume):

    volShape = volume.shape
    canvas = np.zeros((volShape[1],volShape[2],volShape[0]))
    print('Original volume shape is ',volShape, ' but reshaped to ',canvas.shape )
    for z in range(volShape[0]):
        canvas[:,:,z] = volume[z,:,:]
    return canvas
####

def featureReconstruction(featureVect, image, label):
    # Inputs:
    #   featureVect: pandas Dataframe or Series, containing single feature vector (its size must agree with the number of pixels in "label")
    #   image: numpy array, single slice of 2D image
    #   label: numpy array, containing mask annotation for provide image

    imageVector = np.reshape(image,(image.shape[0]*image.shape[1],1))
    labelVector = np.reshape(label,(label.shape[0]*label.shape[1],1))
    print('__________',np.unique(labelVector),np.shape(featureVect),sum(labelVector))

    # plt.imshow(label)
    # plt.show()
    canvas = np.zeros_like(labelVector, float)
    if len(np.unique(labelVector)) > 2:
        print('\n \tProvided label array contains more than 2 labels')
        return []
    elif len(featureVect) != sum(labelVector):
        print('\n \tSize of feature vector must agree with the positive labels in label-mask')
        return []
    print('------------------->',sum(labelVector)/9, len(featureVect))
    idv = 0
    for it in range(len(labelVector)):
        # print(labelVector[it])￼
        if labelVector[it] == 1 and idv<len(featureVect):
            # print(idv)
            # print(featureVect.iloc[idv])
            canvas[it] = featureVect.iloc[idv]
            # print(canvas[it])
            idv += 1
    heatMap = np.reshape(canvas,(image.shape))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.imshow(heatMap, cmap=matplotlib.cm.jet)
    plt.clim(0,0.00025)
    plt.colorbar()
    plt.imshow(image,cmap='gray',alpha=0.5)
    plt.show()


####### Main part  ###################

imageVol1 = sitk.ReadImage('/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Contrast_resampled/4015008250293_Contrast_FULL_AX_LAVA_Not_FS_Not_BH_POST_raw_resampled_trilinear.mha')     # Reading image
imArray1 = sitk.GetArrayFromImage(imageVol1)     # Getting image array
label1 = sitk.ReadImage('/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/contrast_labels_lumen/4015008250293_Contrast_FULL_AX_LAVA_Not_FS_Not_BH_POST_raw_label_resampled_neares_lummen.mha')    # Reading Label mask
labArray1 = sitk.GetArrayFromImage(label1)    # Getting mask array

imVolume1 = sitk2numpy_xyz(imArray1)
labVolume1 = sitk2numpy_xyz(labArray1)

###### image 2
imageVol2 = sitk.ReadImage('/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Contrast_resampled/4015005466564_Contrast_FULL_AX_LAVA_Not_FS_Not_BH_POST_raw_resampled_trilinear.mha')     # Reading image
imArray2 = sitk.GetArrayFromImage(imageVol2)     # Getting image array
label2 = sitk.ReadImage('/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/contrast_labels_lumen/4015005466564_Contrast_FULL_AX_LAVA_Not_FS_Not_BH_POST_label_resampled_neares_lummen.mha')    # Reading Label mask
labArray2 = sitk.GetArrayFromImage(label2)    # Getting mask array

imVolume2 = sitk2numpy_xyz(imArray2)
labVolume2 = sitk2numpy_xyz(labArray2)

# for mz in range(np.shape(labVolume2)[2]):
#     if sum(labVolume2[:,:,mz].ravel()) >0:
#         print(mz, sum(labVolume2[:,:,mz].ravel()))
#         plt.imshow(labVolume2[:,:,mz])
#         plt.show()

Feature_names = pd.read_csv('/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Features/3410000084986150_slc_134_2D_feature_names.csv')


print(imArray2.shape,imVolume2.shape)
featList = os.listdir('/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Features/')
for ft in featList:
    print('.....................',ft)
    if (str(4015008250293) in ft) and (str('feature_values') in ft):
        pieces = ft.split('_')
        sl = int(pieces[2])
        print(sl)
        feature1 = pd.read_csv('/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Features/'+ft)
        print('features:\n',np.absolute(feature1.iloc[:,55]))
        print(labVolume1.shape,np.unique(labVolume1))
        # plt.imshow(labVolume1[:,:,sl])
        # plt.show()
        labVolume1 = np.where(labVolume1==np.unique(labVolume1)[1],1,0)
        print(labVolume1.shape,np.unique(labVolume1))
        print('-------------------x>',Feature_names.iloc[175,1])
        featureReconstruction(feature1.iloc[:,175],imVolume1[:,:,sl],labVolume1[:,:,sl]) # 179, 147
    #Feature_names = pd.read_csv(contrast_path+'3410000084986150_slc_134_2D_feature_stats_names_A.csv')

    elif (str(4015005466564) in ft) and (str('feature_values') in ft):  #4015004554362
        pieces = ft.split('_')
        sl = int(pieces[2])
        print(sl)
        feature2 = pd.read_csv('/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Features/'+ft)
        # print(feature2)
        print(labVolume2.shape,np.unique(labVolume2))
        # plt.imshow(labVolume2[:,:,sl])
        # plt.show()
        labVolume2 = np.where(labVolume2==np.unique(labVolume2)[1],1,0)
        # feat =main2Dfeature(imVolume2[:,:,sl],
        #                             labVolume2[:,:,sl],
        #                             output_path='/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Features/'+ft,
        #                             windows=[3,5],save_features=True)
        print(labVolume2.shape,np.unique(labVolume2))
        print('------------------>',Feature_names.iloc[175,1])
        featureReconstruction(feature2.iloc[:,175],imVolume2[:,:,sl],labVolume2[:,:,sl]) # feature 179, 147
