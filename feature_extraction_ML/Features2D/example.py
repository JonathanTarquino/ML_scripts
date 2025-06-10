from feature_extract_invent import extract2DFeatureInfo
from main_2D_feature import main2Dfeature
import cv2 as cv
import skimage
import numpy as np

# image = cv.imread(input_image_path)
# mk = cv.imread(input_mask_path)
main2Dfeature('/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/b_53__0_image.jpg','/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/b_53__0_mask.jpg','/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/', windows=[3])

# image = cv.imread('/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/b_53__0_image.jpg')
# print(np.shape(image))
# mk = cv.imread('/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/b_53__0_mask.jpg')
# # plt.imshow(image)
# # plt.show()
#
# # plt.imshow(mk)
# # plt.show()
# threshold = skimage.filters.threshold_otsu(mk[:,:,0])
# mk = mk[:,:,0] > threshold
# print('point 1')
#
# all_features = extract2DFeatureInfo(image[:,:,0],mk,[3,5])
# print(all_features[0][0])
