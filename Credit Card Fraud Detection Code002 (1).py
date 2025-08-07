import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Display the class distribution before SMOTE
print("Class distribution before SMOTE:")
print(y.value_counts())

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Display the class distribution after SMOTE
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model
predictions = clf.predict(X_test)
report = classification_report(y_test, predictions)
roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
conf_matrix = confusion_matrix(y_test, predictions)

print("Classification Report:")
print(report)
print(f"\nROC AUC Score: {roc_auc}")

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Fraudulent'], yticklabels=['Legitimate', 'Fraudulent'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 8))
feature_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title('Top 10 Feature Importances')
plt.show()