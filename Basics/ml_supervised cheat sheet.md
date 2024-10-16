Here's a **Supervised Machine Learning & Scikit-learn Cheatsheet** that covers the key steps and functions you’ll typically use in building machine learning models for classification and regression tasks. This cheatsheet is tailored to assist with the workflows of data scientists using scikit-learn for supervised learning.

### **1. Importing Libraries**
```python
# Basic imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
```

---

### **2. Loading and Splitting Data**
```python
# Load data (example using Pandas)
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### **3. Feature Scaling**
```python
# Scale features (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### **4. Classification Algorithms**
#### **Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression

# Create and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
```

#### **Decision Tree Classifier**
```python
from sklearn.tree import DecisionTreeClassifier

# Create and fit the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

#### **Random Forest Classifier**
```python
from sklearn.ensemble import RandomForestClassifier

# Create and fit the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

#### **Support Vector Machine (SVM)**
```python
from sklearn.svm import SVC

# Create and fit the model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

#### **K-Nearest Neighbors (KNN)**
```python
from sklearn.neighbors import KNeighborsClassifier

# Create and fit the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

---

### **5. Regression Algorithms**
#### **Linear Regression**
```python
from sklearn.linear_model import LinearRegression

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R2 Score: {r2}')
```

#### **Decision Tree Regressor**
```python
from sklearn.tree import DecisionTreeRegressor

# Create and fit the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
```

#### **Random Forest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor

# Create and fit the model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
```

#### **Support Vector Regressor (SVR)**
```python
from sklearn.svm import SVR

# Create and fit the model
model = SVR(kernel='linear')
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
```

---

### **6. Model Evaluation Metrics**
#### **Classification Metrics**
```python
# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Precision, Recall, F1-Score
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(cm)
print(report)
```

#### **Regression Metrics**
```python
# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R-squared (R2 Score)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}, RMSE: {rmse}, R2 Score: {r2}')
```

---

### **7. Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Mean of cross-validation scores
print(f'Mean Cross-Validation Score: {cv_scores.mean()}')
```

---

### **8. Hyperparameter Tuning**
#### **Grid Search**
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
print(f'Best parameters: {grid_search.best_params_}')
```

#### **Random Search**
```python
from sklearn.model_selection import RandomizedSearchCV

# Define parameter grid
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=5)

# Fit the model
random_search.fit(X_train, y_train)

# Get the best parameters
print(f'Best parameters: {random_search.best_params_}')
```

---

### **9. Saving & Loading Models**
```python
import joblib

# Save the model to a file
joblib.dump(model, 'model.pkl')

# Load the model from the file
loaded_model = joblib.load('model.pkl')
```

---

### **10. Feature Importance**
```python
# Feature importance for tree-based models
importance = model.feature_importances_

# Plot feature importance
import matplotlib.pyplot as plt
plt.barh(range(len(importance)), importance)
plt.yticks(range(len(importance)), X.columns)
plt.show()
```

This cheatsheet provides a quick overview of the most commonly used techniques and steps in supervised machine learning using **scikit-learn**. It’s designed to help you navigate classification, regression, model evaluation, and tuning.