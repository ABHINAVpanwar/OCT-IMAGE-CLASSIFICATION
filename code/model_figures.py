import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = 'data/merged_file.csv'
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

# Feature importance
feature_importances = rf_model.feature_importances_
features = X_df.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.show()

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_probs)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(False)
plt.show()

# Plot error distribution
errors = y_test - y_test_pred
plt.figure(figsize=(8, 6))
sns.histplot(errors, kde=True, color="coral")
plt.title("Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X_df, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training Score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="green", label="Validation Score")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.title("Learning Curve - Random Forest")
plt.legend(loc="best")
plt.grid(False)
plt.show()
