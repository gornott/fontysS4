import os
import numpy as np
import sys
from skimage.io import imread
from skimage.transform import resize
from skimage.io import imshow
import mayavi.mlab as mlab


image_dir = '/Users/metoditarnev/Desktop/letters/img'
#Get the firts image from the folder
image = imread(os.path.join(image_dir, os.listdir(image_dir)[1]), as_gray=True)
mlab.imshow(image)
#Resize the image

#Get the pixel features
features = np.reshape(image.flatten(), (900*1200))
np.set_printoptions(threshold=sys.maxsize)# shape of feature array
print('\n\nShape of the feature array = ',features.shape)

print(features, '\n\n')




