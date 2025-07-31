import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from feature_extraction_ML.Features3D.itk2numpy_tools import sitk2numpy_xyz
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.morphology import binary_erosion, binary_opening, disk, binary_dilation
from skimage.filters import threshold_otsu, threshold_yen, threshold_li, threshold_minimum, threshold_isodata, threshold_sauvola,threshold_niblack
import os

def get_lumen(image,mask):
    print(image.shape)
    regions = regionprops(mask)
    # plt.imshow(image)
    # plt.imshow(mask,alpha =0.5)
    # plt.show()

    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)

        d = np.uint8(props.area/150)
        print('Area..............',props.area,d)
        footprint0=np.ones((d, d))

        patch = image[minr:maxr,minc:maxc]
        mask_patch = mask[minr:maxr,minc:maxc]
        canvas = np.zeros((mask_patch.shape))
        print('1',maxr-minr,'1',maxc-minc)
        canvas[1:maxr-minr-3,1:maxc-minc-3] =mask_patch[2:maxr-minr-2,2:maxc-minc-2]
        # plt.imshow(patch)
        # plt.show()
        # plt.imshow(canvas)
        # plt.show()

        mask_patch = binary_erosion(canvas,footprint0)
        # plt.imshow(mask_patch)
        # plt.show()
        th = threshold_sauvola(patch)
        # print('............',th)
        lumen = patch < th

        # plt.imshow(lumen)
        # plt.show()

        footprint = disk(1)
        lumen = np.logical_and( mask_patch,lumen)

        lumen = np.reshape(lumen,mask_patch.shape)
        # lumen = binary_opening(lumen,footprint)
        lumen = binary_opening(lumen,footprint)
        lumen = binary_dilation(lumen,np.ones((d,d)))
        lumen = np.logical_and( mask_patch,lumen)
        lumen = np.uint8(lumen)
        # plt.imshow(patch)
        # plt.imshow(lumen, alpha=0.5)
        # plt.show()
        # print('______',np.unique(lumen))
        mask[minr:maxr,minc:maxc]=mask[minr:maxr,minc:maxc]+lumen
        # plt.imshow(mask)
        # plt.show()

    return mask

def segment_lumen(image_path,label_path,output_path):
    images_list = os.listdir(image_path)
    masks_list = os.listdir(label_path)

    for image in images_list:
        im = sitk.ReadImage(image_path+image)
        name = image.split('_')
        imArray = sitk.GetArrayFromImage(im)
        canvas = np.zeros(imArray.shape)
        # print('-------------->',name[0],image)
        for mask in masks_list:
            if name[0] in mask:
                print(mask)
                if '_AX' in mask:
                    # print(mask)
                    labelImage = sitk.ReadImage(label_path+mask)
                    # imArray = sitk2numpy_xyz(imArray)
                    mkArray = sitk.GetArrayFromImage(labelImage)
                    # mkArray = sitk2numpy_xyz(mkArray)
                    values = np.unique(mkArray)
                    print('--------------------->',values)

                    canvas = [np.where(mkArray== np.max(values),np.max(values),np.min(values))]
                    canvas = np.reshape(canvas,mkArray.shape)
                    print(np.unique(canvas),np.unique(values),np.shape(canvas))
                    # plt.imshow(imArray[200,:,:])
                    # plt.imshow(canvas[200,:,:],alpha = 0.35)
                    # plt.show()
                    for idx in range(canvas.shape[0]):
                        if canvas[idx,:,:].any() !=0:
                            print('slice', idx)
                            mkArray[idx,:,:] = get_lumen(imArray[idx,:,:],canvas[idx,:,:])
                else:
                    labelImage = sitk.ReadImage(label_path+mask)
                    mkArray = sitk.GetArrayFromImage(labelImage)
                    values = np.unique(mkArray)
                    print('--------------------->',values)
                    canvas = [np.where(mkArray == np.max(values),np.max(values),np.min(values))]
                    canvas = np.reshape(canvas,mkArray.shape)
                    # plt.imshow(imArray[:,180,:])
                    # plt.imshow(canvas[:,180,:], alpha = 0.5)
                    # plt.show()
                    for idx in range(canvas.shape[1]):
                        if canvas[:,idx,:].any() !=0:
                            print('slice', idx)
                            mkArray[:,idx,:] = get_lumen(imArray[:,idx,:],canvas[:,idx,:])

                mk_Pivot = [np.where(mkArray >= np.max(values),mkArray,0)]
                mk_Pivot = np.reshape(mk_Pivot,mkArray.shape)
                print(np.unique(mk_Pivot),mk_Pivot.shape)

                labelImage = sitk.GetImageFromArray(mk_Pivot)
                print('Writing ...',output_path+mask[0:-5]+'_lummen.mha')
                sitk.WriteImage(labelImage,output_path+mask[0:-5]+'_lummen.mha')



# __main___
volumes = '/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Contrast_resampled/'
labels = '/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Label_resampled/'
out = '/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/contrast_labels_lumen/'
segment_lumen(volumes,labels,out)
