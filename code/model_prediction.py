import cv2
import joblib
import pandas as pd
from utils import compute_first_order_features, compute_glcm_properties

# Load the saved Random Forest model and scaler
rf_model = joblib.load('C:/Users/Abhinav/OneDrive/Documents/MINI_PROJECT/models/random_forest_model.pkl')
scaler = joblib.load('C:/Users/Abhinav/OneDrive/Documents/MINI_PROJECT/models/scaler.pkl')

# Function to process and predict for a single image
def predict_image(image_path):
    image = cv2.imread(image_path)
    
    if image is not None:
        # Resize and compute features
        resized_image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
        first_order_features = compute_first_order_features(resized_image)
        texture_props = compute_glcm_properties(resized_image)
        
        # Combine features into a DataFrame
        features = first_order_features + list(texture_props)
        new_data = pd.DataFrame([features], columns=['mean_intensity', 'std_dev', 'median', 'variance',
                                                     'skewness', 'kurtosis', 'entropy',
                                                     'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'])
        
        # Scale the features
        new_data_scaled = scaler.transform(new_data)
        
        # Make prediction
        prediction = rf_model.predict(new_data_scaled)
        prediction_proba = rf_model.predict_proba(new_data_scaled)
        
        return prediction[0], prediction_proba
    else:
        print(f"Failed to load image: {image_path}")
        return None, None

image_path = "C:/Users/Abhinav/OneDrive/Documents/MINI_PROJECT/dataset/AMD/amd_1047099_1.jpg"
pred_class, pred_proba = predict_image(image_path)
dictionary = {0.0:"Normal",1.0:"AMD"}
print(f"Predicted class: {dictionary[pred_class]}")
print(f"Class probabilities: {pred_proba}")