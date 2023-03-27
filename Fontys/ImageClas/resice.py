import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load images and labels
img_dir = '/Users/metoditarnev/Desktop/letters/img'
df = pd.read_csv('/Users/metoditarnev/Desktop/letters/english.csv')
# Load labels into a list
labels = df['label'].tolist()

# Create a list to store image features and labels
features = []

# Loop over all images in the directory
for filename in os.listdir(img_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load image and convert to grayscale
        img = cv2.imread(os.path.join(img_dir, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute SIFT descriptors
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Add features and label to list
        features.append(descriptors)

# Stack descriptors for all images into a single array
features = np.vstack(features)

# Train a k-means clustering model on the descriptors
k = 100
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, clusters, centers = cv2.kmeans(features, k, None, criteria, 10, flags)

# Compute bag-of-words histograms for each image
histograms = []
for i in range(len(labels)):
    descriptors = features[i * k:(i + 1) * k, :]
    histogram = np.zeros(k, dtype=int)
    for j in range(len(descriptors)):
        distances = np.linalg.norm(centers - descriptors[j, :], axis=1)
        nearest_cluster = np.argmin(distances)
        histogram[nearest_cluster] += 1
    histograms.append(histogram)

# Convert histograms and labels to numpy arrays
X = np.vstack(histograms)
y = np.array(labels)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a support vector machine classifier
from sklearn.svm import SVC
param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 'auto']}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)
best_svm = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)