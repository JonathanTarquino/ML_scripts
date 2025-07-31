import SimpleITK as sitk
import matplotlib.pyplot as plt
import pdb
import os
import numpy as np
from BiasFieldcorr import N4
# Working in Mha images

def builtAlignedVol(original,reference,maskFM,cutType):
    volume = sitk.GetArrayFromImage(reference)
    label = sitk.GetArrayFromImage(original)
    mask = sitk.GetArrayFromImage(maskFM)

    Volspacing = reference.GetSpacing()  # Returns a list of spacing values (e.g., [0.5, 0.5])
    Volorigin = reference.GetOrigin() # Returns the origin coordinates (e.g., [10.0, 20.0])
    Labspacing = reference.GetSpacing()  # Returns a list of spacing values (e.g., [0.5, 0.5])
    Laborigin = reference.GetOrigin() # Returns the origin coordinates (e.g., [10.0, 20.0])

    physicalPointLabel = original.TransformIndexToPhysicalPoint((0,0,0))
    physicalPointVol = reference.TransformIndexToPhysicalPoint((0,0,0))

    # Convert a physical point (10mm, 20mm) to pixel indices
    pL_x = physicalPointLabel[0]-physicalPointVol[0]
    pL_y = physicalPointLabel[1]-physicalPointVol[1]
    pL_z = physicalPointLabel[2]-physicalPointVol[2]
    print('________________D',reference.GetDirection(),original.GetDirection())

    if cutType == 'AX':
        print('xxxxxxxxxxx',label.shape,volume.shape)
        slc = [n for n in range(np.shape(label)[0]) if (label[n,:,:].any()==1)]
        # slc.sort(reverse=True)

        canvasVol = np.zeros((len(slc),volume.shape[1],volume.shape[2]))
        canvasLabel = np.zeros((len(slc),volume.shape[1],volume.shape[2]))
        canvasFM = np.zeros((len(slc),volume.shape[1],volume.shape[2]))
        print(cutType,'............',slc,np.shape(canvasVol))
        newIdx = 0

        for idx in slc:
            physicalPointVol = reference.TransformIndexToPhysicalPoint((idx,0,0))
            physicalPointLabel = original.TransformIndexToPhysicalPoint((idx,0,0))
            # print(reference.TransformPhysicalPointToIndex(physicalPointLabel))
            vol_idx = reference.TransformPhysicalPointToIndex(physicalPointLabel)[0]
            # print(original.TransformPhysicalPointToIndex(physicalPointLabel))
            # label_idx= original.TransformPhysicalPointToIndex(physicalPointLabel)[0]
            pL_x = physicalPointLabel[0]-physicalPointVol[0]
            # print('***************',idx,int(physicalPointLabel[0]),int(physicalPointVol[0]))
            # plt.imshow(volume[vol_idx,:,:])
            # plt.imshow(label[idx,:,:],alpha=0.3)
            # plt.show()
            canvasVol[newIdx,:,:] = volume[vol_idx,:,:]
            canvasLabel[newIdx,:,:] = label[idx,:,:]
            canvasFM[newIdx,:,:] = mask[vol_idx,:,:]
            newIdx += 1
    else:
        print('xxxxxxxxxxx',label.shape,volume.shape)
        slc = [n for n in range(np.shape(label)[0]) if (label[n,:,:].any()==1)]
        # slc.sort(reverse=True)

        canvasVol = np.zeros((len(slc),volume.shape[1],volume.shape[2]))
        canvasLabel = np.zeros((len(slc),volume.shape[1],volume.shape[2]))
        canvasFM = np.zeros((len(slc),volume.shape[1],volume.shape[2]))

        print(cutType,'............',slc,np.shape(canvasVol))
        newIdx = 0

        for idx in slc:
            physicalPointVol = reference.TransformIndexToPhysicalPoint((idx,0,0))
            physicalPointLabel = original.TransformIndexToPhysicalPoint((idx,0,0))
            # print(reference.TransformPhysicalPointToIndex(physicalPointLabel))
            vol_idx = reference.TransformPhysicalPointToIndex(physicalPointLabel)[0]
            # print(original.TransformPhysicalPointToIndex(physicalPointLabel))
            label_idx= original.TransformPhysicalPointToIndex(physicalPointLabel)[0]
            pL_x = physicalPointLabel[0]-physicalPointVol[0]
            # print('***************',idx,int(physicalPointLabel[0]),int(physicalPointVol[0]))
            # plt.imshow(volume[vol_idx,:,:])
            # plt.imshow(label[idx,:,:],alpha=0.3)
            # plt.show()
            canvasVol[newIdx,:,:] = volume[vol_idx,:,:]#4015005459284
            canvasLabel[newIdx,:,:] = label[idx,:,:]
            canvasFM[newIdx,:,:] = mask[vol_idx,:,:]
            newIdx += 1

    plt.imshow(canvasVol[10,:,:])
    plt.imshow(canvasLabel[10,:,:],alpha=0.5)
    plt.imshow(canvasFM[10,:,:],alpha=0.5)
    plt.show()
    return canvasVol,canvasLabel,canvasFM

def cropVolume2VOI(image_path,label_path,mask_path, output_path,cLabel_path):
    patients = [i for i in os.listdir(image_path) if not i.startswith('.') and i.endswith('.mha')]
    masks = [i for i in os.listdir(mask_path) if not i.startswith('.') and i.endswith('.mha')]
    labels = [i for i in os.listdir(label_path) if not i.startswith('.') and i.endswith('.mha')]
    print(patients.sort(),masks.sort(),labels.sort(), patients)
    for pat_id in range(25,len(patients)):
        print(patients[pat_id])
        words = patients[pat_id].split('_')
        print(words[0])
        for label in labels:
            for mask in masks:
                if (words[0] in label) and (words[0] in mask):
                    print(words[0],label,mask)

                    inputImage = sitk.ReadImage(image_path+patients[pat_id])
                    # inputImage.TransformIndexToPhysicalPoint((0,0,0))
                    print('-------------',inputImage.GetOrigin(), inputImage.GetSpacing())
                    spaceIm = inputImage.GetSpacing()
                    orgIm = inputImage.GetOrigin()
                    directIm = inputImage.GetDirection()
                    volumeIm = sitk.GetArrayFromImage(inputImage)
                    print(np.shape(volumeIm))

                    maskImage = sitk.ReadImage(mask_path+mask)
                    print('mask-------------',maskImage.GetOrigin(), maskImage.GetSpacing())
                    spaceMask = maskImage.GetSpacing()
                    orgMask = maskImage.GetOrigin()
                    directMask = maskImage.GetDirection()
                    volumeMask = sitk.GetArrayFromImage(maskImage)
                    print('Mask shape:',np.shape(volumeMask))


                    labelImage = sitk.ReadImage(label_path+label)
                    # labelImage.TransformIndexToPhysicalPoint((0,0,0))
                    print('label-------------',labelImage.GetOrigin(), labelImage.GetSpacing())
                    spacelabel = labelImage.GetSpacing()
                    orglabel = labelImage.GetOrigin()
                    directlabel = labelImage.GetDirection()
                    # labelImage.SetOrigin(orgIm)
                    volumelabel = sitk.GetArrayFromImage(labelImage)
                    print('label shape:',np.shape(volumelabel))

                    sitk.Resample(labelImage, inputImage, sitk.Transform(), sitk.sitkNearestNeighbor, 0, labelImage.GetPixelID())
                    print('--->>>>>>>>>>>>',labelImage.GetOrigin(), labelImage.GetSpacing(), np.shape(sitk.GetArrayFromImage(labelImage)) )

                    if 'AX' in words:
                        nVol,nLab,nFM = builtAlignedVol(labelImage,inputImage,maskImage,'AX')
                    else:
                        nVol,nLab,nFM = builtAlignedVol(labelImage,inputImage,maskImage,'COR')

                    print('Cropped image shapes:...............',nVol.shape,nLab.shape,nFM.shape)
                    inputImage = sitk.GetImageFromArray(nVol)
                    labelImage = sitk.GetImageFromArray(nLab)
                    maskImage = sitk.GetImageFromArray(nFM)


                    print('Writing ...',cropLabeldir+label[0:-5]+'_cropped.mha')
                    sitk.WriteImage(labelImage,cropLabeldir+label[0:-5]+'_cropped.mha')
                    # print(we)

                    N4(inputImage,maskImage,output_path+'_'+patients[pat_id][0:-5]+'_BFC.mha')




#main

# root = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/T2_40P_resampled/'
root = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Contrast_resampled/'
# maskdir = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/T2_40P_resampled_foreground_masks/'  # foreground masks that define region you want to apply BFC on
maskdir = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Contrast_resampled_ForegroundMask/'
root_out = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Contrast_resampled_Croped_BFC/'
# labeldir = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/labels/'
labeldir = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Label_resampled/'
cropLabeldir = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/final_labels/'

cropVolume2VOI(root,labeldir,maskdir,root_out,cropLabeldir)
