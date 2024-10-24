import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = 'C:/Users/Abhinav/OneDrive/Documents/MINI_PROJECT/data/merged_file.csv'
data = pd.read_csv(df)

# Select relevant features
selected_features = ['mean_intensity', 'std_dev', 'median', 'variance', 'skewness', 'kurtosis', 'entropy',
                     'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'Target']
data_selected = data[selected_features]

# Check for missing values in y (Target)
missing_y = data_selected[data_selected["Target"].isnull()]
print(f"Rows with missing 'Target':\n{missing_y}")
print(f"Number of missing values in 'Target': {data_selected['Target'].isnull().sum()}")

# Drop rows with missing Target values (optional, based on your approach)
data_selected = data_selected.dropna(subset=["Target"])

X = data_selected.drop(["Target"], axis=1)
y = data_selected["Target"]

# Standardize the numeric features
X_numeric = X.select_dtypes(include=np.number)
s_scaler = preprocessing.StandardScaler()
X_scaled = s_scaler.fit_transform(X_numeric)
X_df = pd.DataFrame(X_scaled, columns=X_numeric.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict and calculate accuracy
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print('Confusion matrix \n', cm)

# Plot confusion matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Class 0", "Class 1"],
            yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Classification report
report = classification_report(y_test, y_test_pred)
print("\nClassification Report:\n", report)

# ROC and AUC
y_probs = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
auc = roc_auc_score(y_test, y_probs)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.grid(False)
plt.show()