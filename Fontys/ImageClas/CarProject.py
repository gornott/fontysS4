import numpy as np
from skimage.io import imread
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the training images and labels
# Load the training images and labels
df = pd.read_csv('/Users/metoditarnev/Desktop/letters/english.csv')

# Preprocess the images and extract HOG features
image_dir = '/Users/metoditarnev/Desktop/letters/img'

train_images = []
train_labels = df['label']
for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image = imread(os.path.join(image_dir, filename))
        if image is not None:
            train_images.append(image.flatten())
train_images = np.array(train_images, dtype=object)
train_images = train_images / 255.0

# train_features = np.array(train_features)
x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, shuffle=True)
# Train a classifier
clf = svm.SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(clf, parameters)
clf.fit(x_train, y_train)
best_estimator = grid_search.best_estimator_
# Predict the test set
y_pred = best_estimator.predict(x_test)
# Evaluate the classifier
score = accuracy_score(y_test, y_pred)
print('{}% of samples were correctly classified'.format(str(score * 100)))
