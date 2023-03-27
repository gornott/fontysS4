import os
import pandas as pd
from skimage.io import imread
import csv

# Set the path to your image directory and label file
image_dir = '/Users/metoditarnev/Desktop/letters/img'
label_file = '/Users/metoditarnev/Desktop/letters/english.csv'

# Load the labels into a pandas dataframe
labels = pd.read_csv(label_file)

# Open a CSV file for writing
with open('pixel_values.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Loop through each image file
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            # Read in the image and flatten it
            image = imread(os.path.join(image_dir, filename), as_gray=True)
            label = labels.loc[labels['image'] == filename, 'label'].iloc[0]
            flattened_image = image.flatten()

            # Write the pixel values to the CSV file along with the label
            writer.writerow(list(flattened_image) + [label])
