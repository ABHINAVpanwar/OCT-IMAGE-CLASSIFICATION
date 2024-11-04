import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('data/merged_file.csv')

# Selecting features and target
selected_features = ['mean_intensity', 'std_dev', 'median', 'variance', 'skewness', 'kurtosis', 'entropy',
                     'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'Target']
data_selected = df[selected_features].dropna()

X = data_selected.drop(['Target'], axis=1)
y = data_selected['Target']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model and the scaler
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Evaluation
train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")
