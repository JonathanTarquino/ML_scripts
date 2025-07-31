#https://github.com/bigbigbean/N4BiasFieldCorrection
#https://github.com/bigbigbean/N4BiasFieldCorrection
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pdb
import os
import numpy as np


def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.mha':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def N4(inputfile,maskfile,outputfile):
    print("N4 bias correction runs.")
    if isinstance(inputfile, str):
        print('...from dir')
        inputImage = sitk.ReadImage(inputfile)
        maskImage = sitk.ReadImage(maskfile, sitk.sitkUInt8)
        #maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
        #sitk.WriteImage(maskImage, "06-t1c_mask3.nii")
        # print(inputImage.GetOrigin(),maskImage.GetOrigin())
        # volume = sitk.GetArrayFromImage(inputImage)
        # print("N4 bias correction runs...")
        # Mak = sitk.GetArrayFromImage(maskImage)
        # print("N4 bias correction runs.....")
        # # print(np.shape(volume),np.shape(maskImage))
        #
        # plt.imshow(volume[60,:,:])
        # plt.imshow(Mak[60,:,:], alpha = 0.7)
        # plt.show()
        inputImage = sitk.Cast(inputImage,sitk.sitkFloat64)
        maskImage = sitk.Cast(maskImage,sitk.sitkUInt8)
        corrector = sitk.N4BiasFieldCorrectionImageFilter();
        print('Building corrector...')
        output = corrector.Execute(inputImage,maskImage)
        print("Finished N4 Bias Field Correction.....")
        sitk.WriteImage(output,outputfile)

    else:
        print('...from file')
        inputImage = sitk.Cast(inputfile,sitk.sitkFloat64)
        maskImage = sitk.Cast(maskfile,sitk.sitkUInt8)
        corrector = sitk.N4BiasFieldCorrectionImageFilter();
        print('Building corrector...')
        output = corrector.Execute(inputImage,maskImage)
        print("Finished N4 Bias Field Correction...")
        sitk.WriteImage(output,outputfile)
        print("Finished N4 Bias Field Correction.....")


if __name__=='__main__':
   root = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/T2_40P_resampled/'
   maskdir = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/T2_40P_resampled_foreground_masks'  # foreground masks that define region you want to apply BFC on
   root_out = r'/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/T2_40P_resampled_BFC/'
   if not os.path.exists(root_out):
    os.makedirs(root_out)

   patients = [i for i in os.listdir(root) if not i.startswith('.') and i.endswith('.mha')]
   masks = [i for i in os.listdir(maskdir) if not i.startswith('.') and i.endswith('.mha')]
   print(patients.sort(),masks.sort())

   for idx in range(0,len(patients)):
      filename=root + os.sep + patients[idx]
      maskname=maskdir + os.sep + masks[idx]
      print(filename,'\n',maskname)
      _, base, _ = split_filename(filename)
      print('------------->', base)
      file_out=root_out + os.sep + base + '.mha'
      N4(filename,maskname,file_out)


# from __future__ import print_function
#
# import SimpleITK as sitk
# import sys
# import numpy as np
# import os
# import glob
# import matplotlib.pyplot as plt
#
# def N4BFC(inputImage,outputImage):
#     if len ( sys.argv ) < 2:
#         print( "Usage: N4BiasFieldCorrection inputImage " + \
#             "outputImage [shrinkFactor] [maskImage] [numberOfIterations] " +\
#             "[numberOfFittingLevels]" )
#         sys.exit ( 1 )
#
#
#     imageName = os.path.split(sys.argv[1])[1]
#     print(imageName)
#
#     inputImage = sitk.ReadImage( sys.argv[1] )
#
#     nda = sitk.GetArrayFromImage(inputImage)
#     plt.imshow(nda[100,:,:])
#     plt.show()
#
#     if len ( sys.argv ) > 4:
#         maskImage = sitk.ReadImage( sys.argv[4] )
#     else:
#         maskImage = sitk.OtsuThreshold( inputImage, 0, 1, 200 )
#
#
#     inputImage = sitk.Cast( inputImage, sitk.sitkFloat32 )
#
#     corrector = sitk.N4BiasFieldCorrectionImageFilter();
#
#     numberFilltingLevels = 4
#
#     if len ( sys.argv ) > 6:
#         numberFilltingLevels = int( sys.argv[6] )
#
#     if len ( sys.argv ) > 5:
#         corrector.SetMaximumNumberOfIterations( [ int( sys.argv[5] ) ] *numberFilltingLevels  )
#
#
#     output = corrector.Execute( inputImage, maskImage )
