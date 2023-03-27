import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.transform import resize
from skimage.feature import hog
from skimage.feature import sift
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/metoditarnev/Desktop/letters/english.csv')

# Preprocess the images and extract HOG features
image_dir = '/Users/metoditarnev/Desktop/letters/img'

# Extract Hog features
# train_features = []
# train_labels = df['label']
# for filename in os.listdir(image_dir):
#     if filename.endswith(".png"):
#         image = imread(os.path.join(image_dir, filename))
#         if image is not None:
#             image = rgb2gray(image)
#             image = resize(image, (100, 100))
#             fd = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
#             train_features.append(fd)
# train_features = np.array(train_features)
# train_features = train_features / 255.0
# Extract SIFT features
train_features = []
train_labels = df['label']
for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image = imread(os.path.join(image_dir, filename))
        if image is not None:
            image = rgb2gray(image)
            image = resize(image, (100, 100))
            fd = sift
            train_features.append(fd)
train_features = np.array(train_features)
train_features = train_features / 255.0



# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(train_features, df['label'], test_size=0.2, random_state=42)

# Train a support vector machine classifier
param_grid = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

# Create an SVM model
clf = SVC()

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=10, verbose=3, n_jobs=-1)
grid_search.fit(x_train, y_train)
best = grid_search.best_estimator_

# Evaluate the model with the best hyperparameters on the test set
# clf_best = SVC(kernel='rbf', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
# clf_best.fit(x_train, y_train)
y_pred = best.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)



