from sklearn import datasets
from skimage.io import imread
import os
digits = datasets.load_digits()
print(digits.data[0])
imput_dir = '/Users/metoditarnev/Downloads/archive (2)/train_v2/train'
image = imread(imput_dir)
