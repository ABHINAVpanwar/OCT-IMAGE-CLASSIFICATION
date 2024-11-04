from flask import Flask, jsonify, request
import cv2
from flask_cors import CORS
import joblib
import pandas as pd
import os
import base64
import tempfile

# Insert the path to your utils directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../code')))
from utils import compute_first_order_features, compute_glcm_properties

app = Flask(__name__)
CORS(app, resources={r"/get_result": {"origins": "https://oct-image-classification.netlify.app"}})

# Load the saved Random Forest model and scaler
rf_model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Function to process and predict for a single image
def predict_image(image):
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

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Use a temporary file to store the image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        file.save(temp_file.name)
        image = cv2.imread(temp_file.name)

    if image is None:
        return jsonify({'error': 'Could not read the image'}), 400

    # Predict the class and probabilities
    pred_class, pred_proba = predict_image(image)
    dictionary = {0.0: "Normal", 1.0: "AMD"}

    if pred_class is not None:
        # Save the image to a temporary location
        cv2.imwrite("application/bin/temp_image.jpg", image)

        # Store result in a text file in the app folder
        with open("application/bin/prediction_result.txt", "w") as f:  # Save in the app directory
            f.write(f"{dictionary[pred_class]}\n")
            f.write(f"{pred_proba.tolist()}\n")

        return jsonify({'success': True}), 200
    else:
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/get_result', methods=['GET'])
def get_result():
    # Check for the result file in the app folder
    if os.path.exists("application/bin/prediction_result.txt"):
        with open("application/bin/prediction_result.txt", "r") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                predicted_class = lines[0].strip()
                class_probabilities = list(map(float, lines[1].strip()[1:-1].split(',')))  # Convert string to list safely
                
                # Read the image file and encode it as base64
                with open("application/bin/temp_image.jpg", "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')  # Encode to base64

                return jsonify({
                    'predicted_class': predicted_class,
                    'class_probabilities': class_probabilities,
                    'image': img_data  # Send base64 encoded image data
                })
    return jsonify({'error': 'No results found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
