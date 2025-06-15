from feature_extract_invent import extract2DFeatureInfo
from main_2D_feature import main2Dfeature
import cv2 as cv
import skimage
import numpy as np

#### First way to use feature extraction function
# main2Dfeature('/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/b_53__0_image.jpg','/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/b_53__0_mask.jpg','/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/', windows=[3])


#### Second way to use feature extraction function
image = cv.imread('/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/b_53__0_image.jpg')
print(np.shape(image))
mk = cv.imread('/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/b_53__0_mask.jpg')
#
all_features = main2Dfeature(image[:,:,0],mk[:,:,0],output_path='/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/',windows=[3,5])
# print(all_features[0][0])
