import numpy as np


def sitk2numpy_xyz(volume):

    volShape = volume.shape
    canvas = np.zeros((volShape[1],volShape[2],volShape[0]))
    print('Original volume shape is ',volShape, ' but reshaped to ',canvas.shape )
    for z in range(volShape[0]):
        canvas[:,:,z] = volume[z,:,:]
    return canvas

