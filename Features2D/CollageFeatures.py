from medviz.feats import collage
from medviz.feats.collage.main import Collage, HaralickFeature
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
from LawsFeatures import _image_xor, lte_measures
from skimage.filters import gabor_kernel
from pyfeats import lte_measures

print(Collage)

def compute_collage2d(
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

    try:
        collage = Collage(
            image,
            mask,
            svd_radius=5,
            verbose_logging=True,
            num_unique_angles=64,
            haralick_window_size=haralick_windows,
        )

        collage_feats = collage.execute()

        #print(collage_feats.shape)

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

    return feats,collage_feats,descriptors
