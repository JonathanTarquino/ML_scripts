import os
import sys
import numpy as np
import glob
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from main_2D_feature import main2Dfeature
# from feature_extraction_ML.Features2D.feature_extract_invent import extract2DFeatureInfo



def sitk2numpy_xyz(volume):

    volShape = volume.shape
    canvas = np.zeros((volShape[1],volShape[2],volShape[0]))
    print('Original volume shape is ',volShape, ' but reshaped to ',canvas.shape )
    for z in range(volShape[0]):
        canvas[:,:,z] = volume[z,:,:]
    return canvas
####
# print(os.path.abspath(__file__))
# sys.path.append(module_path)


####
dataframe = pd.read_excel('/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI_Sheba_groupings.xlsx', sheet_name='Contrast')
print(dataframe)
df_time2surg = dataframe['time diff surgery']
print(df_time2surg.dtypes)
df_filter = dataframe[df_time2surg<=9]
print('........',len(df_filter))

image_folder = '/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/Contrast_resampled/'
Label_folder = '/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/contrast_labels_lumen/'
listIM = os.listdir(image_folder)
listLab = os.listdir(Label_folder)


time2recurrence = []
noLumenCases = []
withLumenCases = []
noLumenFeatures = []
noLumenFeatures = pd.DataFrame (noLumenFeatures)
LumenFeatures = []
LumenFeatures = pd.DataFrame(LumenFeatures)

count = 0
for idx in df_filter['Patient']:
    for im in listIM:
        for lab in listLab:
            if (str(idx) in im) and (str(idx) in lab):

                count+=1
                print(idx,im,lab,'...................',count)
                image = sitk.ReadImage(image_folder+im)     # Reading image
                imArray = sitk.GetArrayFromImage(image)     # Getting image array
                label = sitk.ReadImage(Label_folder+lab)    # Reading Label mask
                labArray = sitk.GetArrayFromImage(label)    # Getting mask array

                imVolume = sitk2numpy_xyz(imArray)
                labVolume = sitk2numpy_xyz(labArray)

                labValues = np.unique(labVolume.flatten())
                areaNolumen = 0
                areaWithlumen = 0
                slcNolumen = 0
                slcWithlumen = 0
                for idz in range(labVolume.shape[2]):

                    if len(np.unique(labVolume[:,:,idz])) == 2:
                        print(np.unique(labVolume[:,:,idz]))

                        if sum(labVolume[:,:,idz].flatten())> areaNolumen:
                            areaNolumen = sum(labVolume[:,:,idz].flatten())
                            slcNolumen = idz

                    elif len(np.unique(labVolume[:,:,idz])) == 3:
                        print(np.unique(labVolume[:,:,idz]))
                        if sum(labVolume[:,:,idz].flatten())> areaWithlumen:
                            areaWithlumen = sum(labVolume[:,:,idz].flatten())
                            print('----->',areaWithlumen,slcWithlumen)
                            slcWithlumen = idz

                print('oooooooooooooooooooooooooooooooooooo>',slcWithlumen,slcNolumen)
                if slcNolumen != 0:
                    # plt.imshow(labVolume[:,:,slcNolumen])
                    # plt.show()
                    noLumenCases.append(idx)



                    print('cases without lumen.....................',noLumenCases,)
                    feat =main2Dfeature(imVolume[:,:,slcNolumen],
                                  labVolume[:,:,slcNolumen],
                                  output_path='/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/',
                                  windows=[3,5,7,9,11],save_features=False)
                    print('features',np.shape(feat[2]))
                    pd.DataFrame(feat[3]).to_csv(image_folder+'LumenFeatures_names.csv')
                    noLumenFeatures = pd.concat([noLumenFeatures,pd.DataFrame(np.reshape(feat[2],(1,len(feat[2]))))],axis=0)
                    print('No lumen features ***************************************************************************************',noLumenFeatures)

                if slcWithlumen != 0:
                    time2recurrence.append(df_filter[df_filter['Patient']==idx]['time diff clinical recurrence'])
                    print('>>>>>>>>>>>>>>>>>>\n',time2recurrence)
                    # plt.imshow(labVolume[:,:,slcWithlumen])
                    # plt.show()
                    withLumenCases.append(idx)
                    print('Cases with lumen.............',withLumenCases)
                    # feat = main2Dfeature(imVolume[:,:,slcWithlumen],
                    #               labVolume[:,:,slcWithlumen],
                    #               output_path='/home/jonathan/Documents/Invent/Crohns/Sheba_DB/MRI/Final_Preprocessed/',
                    #               windows=[3,5,7,9,11],save_features=False)
                    # print('features',np.shape(feat[2]))
                    # LumenFeatures = pd.concat([LumenFeatures,pd.DataFrame(np.reshape(feat[2],(1,len(feat[2]))))],axis=0)
                    # print('With lumen features ******************************************************************************************',LumenFeatures)

print(count)
#
# LumenFeatures.to_csv(image_folder+'LumenFeatures.csv')
# noLumenFeatures.to_csv(image_folder+'noLumenFeatures.csv')
# withLumenCases = pd.DataFrame(withLumenCases)
# noLumenCases = pd.DataFrame(noLumenCases)
# withLumenCases.to_csv(image_folder+'withLumenCases.csv')
# noLumenCases.to_csv(image_folder+'noLumenCases.csv')

                # main2Dfeature(image[:,:,0],mk[:,:,0],output_path='/home/jonathan/Documents/Invent/Code_traslation/b_53_/b_53__/',windows=[3,5])

