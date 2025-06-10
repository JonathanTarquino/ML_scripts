from feature_extract_invent3D import extract3DFeatureInfo
from main_3D_feature import main3Dfeature
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pdb
import glob
import os
import numpy as np
import cv2 as cv
import skimage

# Set the path of the folder where the  volumes are stored
volumesPath = '/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Contrast_resampled/'
LabelsPath = '/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Label_resampled/'

Vol_list = os.listdir(volumesPath)
Label_list = os.listdir(LabelsPath)
for path in Vol_list:
    print(path)
    patNumber = path.split('_')
    for lab in Label_list:
        print(patNumber[0],type(lab))
        if patNumber[0] in lab:
            main3Dfeature(volumesPath+path,LabelsPath+lab,volumesPath)

