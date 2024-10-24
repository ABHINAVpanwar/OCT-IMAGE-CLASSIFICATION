OCT Classification Web Application
This project is a web application designed to classify Optical Coherence Tomography (OCT) images,
specifically focused on detecting abnormalities such as AMD (Age-related Macular Degeneration) versus Normal cases.
The model utilizes a Random Forest algorithm to make predictions based on the input images.

How to Use
Clone the Repository:

bash
Copy code
git clone <repository_url>
Set Up the Dataset:

Inside the dataset/ folder, create two subfolders:
amd/ for images labeled as AMD.
normal/ for images labeled as Normal.
Add the respective images to these folders.
Run the Web Application:

Navigate to the application/ directory and run the Flask server:
bash
Copy code
cd application
python server.py
Access the Web Interface:

Open a browser and go to http://localhost:5000.
Upload an OCT image, and the model will predict whether the image belongs to the "AMD" or "Normal" class.
Model Performance
Training Accuracy: 1.0
Testing Accuracy: 0.8045
