# ML_scripts
This repository contains a set of scripts for extracting features, 2D or 3D, and perform ML model training and validation in python. Some other tools were also included for reconstructing 2D features in heatmaps, to map simpleITK volume structure from zyx to xyz, among others.

## Feature extraction
This repo provide a complete example of how to use both 2D and 3D fetaure extraction functions, in script ```example.py``` for 2D and ```example3DfeatureExtract.py``` for 3D. Those scripts will give you an idea of how the main functions are working and how to deal with different types of images.

In the case of 3D, the function that deals with the extraction is ```extract3DFeatureInfo``` which is in script named ```feature_extract_invent3D.py```:

```python
def extract3DFeatureInfo(img_org,mask_org,ws_options=[3,5,7,9,11],class_options = ['raw','gray','gradient','haralick','gabor','laws','collage'], with_stats=True):
  # im_org =        numpy array (3D or 2D), containing the original image array
  # mask_org =      numpy array (3D or 2D), containing the original mask array
  # class_options = (OPTIONAL) array of strings corresponding to desired feature classes:
  #                 DEFAULT: class_options = ['raw','gray','gradient','haralick','gabor','laws','collage'] a list like variable containing feature familly names to be extracted
  # ws_options = (  OPTIONAL) array of integers corresponding to desired window levels:
  #                 DEFAULT: ws_options = [3, 5, 7, 9, 11]
  # with_stats =    ( OPTIONAL) logic, if True it set the function to provide feature statistics togheter with the pixel wise feature information
  #                 DEFAULT: tRUE
```
For 2D, the function in charge of extracting features is ```extract2DFeatureInfo``` which is in script named ```feature_extract_invent.py```:

```python
def extract2DFeatureInfo(img_org,mask_org,ws_options=[3,5,7,9,11],class_options = ['raw','gray','gradient','haralick','gabor','laws','collage'], with_stats=True):

  # im_org =        numpy array (3D or 2D), containing the original image array
  # mask_org =      numpy array (3D or 2D), containing the original mask array
  # class_options = (OPTIONAL) array of strings corresponding to desired feature classes:
  #                 DEFAULT: class_options = ['raw','gray','gradient','haralick','gabor','laws','collage'] ### still unused
  # ws_options = (  OPTIONAL) array of integers corresponding to desired window levels:
  #                 DEFAULT: ws_options = [3, 5, 7, 9, 11]
  # with_stats =    ( OPTIONAL) logic, if True it set the function to provide feature statistics togheter with the pixel wise feature information
  #                 DEFAULT: tRUE

```
## Classification, training and cross Validation

This repo also include some scripts for 


## Important!!
To run every function, from outer folders or projects, you should include the actual path of the container script. For example, if you are expecting to run ```extract3DFeatureInfo``` you will have to add the following line at the begining of your script ```sys.path.append('/container_path/Features3D/')```
