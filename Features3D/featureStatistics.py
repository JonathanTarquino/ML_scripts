from feature_extract_invent3D import extract3DFeatureInfo
from scipy.stats import kurtosis, skew
import cv2 as cv
import skimage
import numpy as np
import pandas as pd
import os

def feature_stats(featMatrix):
    fStats = np.zeros((4,np.shape(featMatrix)[1]))
    means = featMatrix.mean(axis = 0)
    sDev = featMatrix.std(axis = 0)
    k = featMatrix.kurtosis(axis = 0)
    sk = featMatrix.skew(axis = 0)
    fStats[0,:] = means
    fStats[1,:] = sDev
    fStats[2,:] = k
    fStats[3,:] = sk
    print(fStats)
    print('=============================== FEATURE STATS ===========================\n',fStats)
    return fStats

