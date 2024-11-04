data_folder = 'dataset'

import numpy as np
import pandas as pd
import ydata_profiling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
import yellowbrick
import joblib

from ydata_profiling import ProfileReport
from pywaffle import Waffle
from statsmodels.graphics.gofplots import qqplot
from PIL import Image
from highlight_text import fig_text
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC, ConfusionMatrix
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.contrib.wrapper import wrap
from yellowbrick.style import set_palette
import cv2
from scipy.stats import skew, kurtosis, entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def compute_first_order_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flattened_image = gray_image.flatten()
    mean_intensity = np.mean(flattened_image)
    std_dev = np.std(flattened_image)
    median = np.median(flattened_image)
    variance = np.var(flattened_image)
    skewness_value = skew(flattened_image)
    kurtosis_value = kurtosis(flattened_image)
    entropy_value = entropy(np.histogram(flattened_image, bins=256)[0])
    return [mean_intensity, std_dev, median, variance, skewness_value, kurtosis_value, entropy_value]

def compute_glcm_properties(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return contrast, dissimilarity, homogeneity, energy, correlation

target_size = (100, 100)

features_list = []
class_labels = []
filenames = []

for class_folder in os.listdir(data_folder):
    class_path = os.path.join(data_folder, class_folder)

    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        image_path = os.path.join(class_path, file)
        image = cv2.imread(image_path)

        if image is not None:
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

            first_order_features = compute_first_order_features(resized_image)
            texture_props = compute_glcm_properties(resized_image)

            features = first_order_features + list(texture_props)
            features_list.append(features)
            class_labels.append(class_folder)
            filenames.append(file)
        else:
            print(f"Failed to load image: {image_path}")

columns = ['mean_intensity', 'std_dev', 'median', 'variance',
           'skewness', 'kurtosis', 'entropy',
           'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
data = pd.DataFrame(data=features_list, columns=columns)
data['filename'] = filenames
data['class_label'] = class_labels


class_mapping = {
    'Normal': 0.0,
    'AMD': 1.0      
}

data['Target'] = data['class_label'].map(class_mapping)

print(data)

output_csv_path = 'data/merged_file.csv'
data.to_csv(output_csv_path, index=False)
print(f"Data saved to {output_csv_path}")
