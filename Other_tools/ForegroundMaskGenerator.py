"""
Created on Mon May 23 14:23:06 2022

By: Amir R. Sadri (ars329@case.edu)
"""

import nibabel as nib
from skimage import exposure as ex
import SimpleITK as sitk
import numpy as np
from skimage.morphology import convex_hull_image
from skimage.filters import threshold_otsu
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
# foreground and background function
def foreground(img):
    try:
        # plt.imshow(img)
        # plt.show()
        h = ex.equalize_hist(img[:,:])*255
        oi = np.zeros_like(img, dtype=np.uint16)
        oi[(img > threshold_otsu(img)) == True] = 1
        oh = np.zeros_like(img, dtype=np.uint16)
        oh[(h > threshold_otsu(h)) == True] = 1
        nm = img.shape[0] * img.shape[1]
        w1 = np.sum(oi)/(nm)
        w2 = np.sum(oh)/(nm)
        ots = np.zeros_like(img, dtype=np.uint16)
        new =( w1 * img) + (w2 * h)
        ots[(new > threshold_otsu(new)) == True] = 1
        conv_hull = convex_hull_image(ots)
        ch = np.multiply(conv_hull, 1)
        fore_image = ch * img
        back_image = (1 - ch) * img
    #     plt.imshow(conv_hull)
    #     plt.show()
    except Exception:
        print('Exception')
        fore_image = img.copy()
        back_image = np.zeros_like(img, dtype=np.uint16)
        conv_hull = np.zeros_like(img, dtype=np.uint16)
    return fore_image, back_image, conv_hull, img[conv_hull], img[conv_hull==False]

##### SET THESE LINES  ####################################################################################

# main directory for reading data
root = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Contrast_resampled/'
output_folder = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Contrast_resampled_ForegroundMask/'  # output directory for saving foreground mask


###########################################################################################################



patients = [i for i in os.listdir(root) if not i.startswith('.') and i.endswith('.mha')]


for file in patients:
    print(file)
    fileSplit = os.path.split(file)
    name = fileSplit[1][0:-4]
    print(name)
    inputImage = sitk.ReadImage(root+file)
    print('-------------',inputImage.GetOrigin(), inputImage.GetSpacing())
    space = inputImage.GetSpacing()
    org = inputImage.GetOrigin()
    direct = inputImage.GetDirection()
    volume = sitk.GetArrayFromImage(inputImage)
    print(np.shape(volume))
    volumeShape = np.shape(volume)
    minAxis = np.argmin(volumeShape)
    print(minAxis)
    # data = nib.load(root + os.sep + file)
    # volume = data.get_fdata()
    mask_volume = np.zeros_like(volume, dtype=np.uint16)
    #os.makedirs(output_folder + os.sep + file[:-4] + 'originad_masks.png')

    # for i in range(np.shape(volume)[2]):
    # # for i in range(1):
    #     img = volume[:,:,i]
    #     fore = foreground(img)[2]
    #     fore = np.multiply(fore, 1)
    #     mask_volume[:,:,i] = fore



    for i in range(np.shape(volume)[minAxis]):
        if minAxis == 0:
            # print('Coronal')
            img = volume[i,:,:]
            fore = foreground(img)[2]
            fore = np.multiply(fore, 1)
            mask_volume[i,:,:] = fore


        elif minAxis == 1:
            print('Axial')
            img = volume[:,i,:]
            fore = foreground(img)[2]
            fore = np.multiply(fore, 1)
            mask_volume[:,i,:] = fore

        else:
            print('this')
            img = volume[:,:,i]
            fore = foreground(img)[2]
            fore = np.multiply(fore, 1)
            mask_volume[:,:,i] = fore


    print('mask shape...',np.shape(mask_volume))
    outputImageFileName = output_folder+name+'_foreMask.mha'
    inputImage = sitk.GetImageFromArray(mask_volume)
    Mask = sitk.GetImageFromArray(mask_volume)

    print('...................',org,direct)

    Mask.SetSpacing(space)
    Mask.SetOrigin(org)
    Mask.SetDirection(direct)
    print(space)
    print('Writing ...',outputImageFileName)
    sitk.WriteImage(Mask,outputImageFileName)
    print('Writing .....',outputImageFileName)
    # # writer = sitk.ImageFileWriter()
    # # writer.SetFileName(outputImageFileName)
    # # writer.Execute(mask_volume)
    # save_image = nib.Nifti1Image(volume, np.eye(4))
    # save_image.to_filename(root + os.sep + '{}.nii'.format(os.path.split(file)[1][:-4]))
    # save_mask = nib.Nifti1Image(mask_volume, np.eye(4))
    # save_mask.to_filename(output_folder + os.sep + '{}_foreground_mask.nii'.format(os.path.split(file)[1][:-4]))







# FB.py
# Displaying FB.py.
