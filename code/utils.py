import cv2
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from skimage.feature import graycomatrix, graycoprops

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