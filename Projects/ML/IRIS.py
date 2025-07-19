# iris_classification.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# Load data
columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class labels']
path = "/Users/abassmac/downloads/IRIS.csv"
df = pd.read_csv(path, names=columns)


# Convert numeric columns and clean data
for col in df.columns[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

# Optional: Visualize data
def plot_data():
    sns.pairplot(df, hue='class labels')
    plt.title("Iris Dataset Pairplot")
    plt.show()

# plot_data()  # Uncomment to see visualization

# Encode labels
le = LabelEncoder()
df['class labels'] = le.fit_transform(df['class labels'])

# Split features and target
X = df.drop('class labels', axis=1)
y = df['class labels']

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

# ---- Model 1: SVM ----
model_svc = SVC()
model_svc.fit(x_train, y_train)
pred_svc = model_svc.predict(x_test)
score_svc = accuracy_score(y_test, pred_svc) * 100
print(f"SVC Accuracy: {score_svc:.2f}%")

# ---- Model 2: Logistic Regression ----
model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)
pred_lr = model_lr.predict(x_test)
score_lr = accuracy_score(y_test, pred_lr) * 100
print(f"Logistic Regression Accuracy: {score_lr:.2f}%")

# ---- Model 3: Decision Tree ----
model_dt = DecisionTreeClassifier()
model_dt.fit(x_train, y_train)
pred_dt = model_dt.predict(x_test)
score_dt = accuracy_score(y_test, pred_dt) * 100
print(f"Decision Tree Accuracy: {score_dt:.2f}%")

# Classification report for Logistic Regression
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, pred_lr, target_names=le.classes_))
