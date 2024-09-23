
# Random Forest in machine learning


- The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. **It generally has much better predictive accuracy than a single decision tree and it works well with default parameters.** If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters.

## What is the Random Forest Algorithm?

Random Forest algorithm is a powerful tree learning technique in Machine Learning. It works by creating a number of Decision Trees during the training phase. Each tree is constructed using a random subset of the data set to measure a random subset of features in each partition. This randomness introduces variability among individual trees, reducing the risk of overfitting and improving overall prediction performance.

In prediction, the algorithm aggregates the results of all trees, either by voting (for classification tasks) or by averaging (for regression tasks) This collaborative decision-making process, supported by multiple trees with their insights, provides an example stable and precise results. Random forests are widely used for classification and regression functions, which are known for their ability to handle complex data, reduce overfitting, and provide reliable forecasts in different environments.

### How it works

- Random forest algorithms have **three main hyperparameters**, which need to be set before training. These include <U>node size, the number of trees, and the number of features sampled</U>. From there, the random forest classifier can be used to solve for regression or classification problems.

- The random forest algorithm is <U>made up of a collection of decision trees, and each tree in the ensemble is comprised of a data sample drawn from a training set with replacement, called the bootstrap sample. Of that training sample, one-third of it is set aside as test data, known as the out-of-bag (oob) sample, which we’ll come back to later.</U> Another instance of randomness is then injected through feature bagging, adding more diversity to the dataset and reducing the correlation among decision trees. Depending on the type of problem, the determination of the prediction will vary. For a regression task, the individual decision trees will be averaged, and for a classification task, a majority vote—i.e. the most frequent categorical variable—will yield the predicted class. Finally, the oob sample is then used for cross-validation, finalizing that prediction.

![randowforest](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/50/f9/ICLH_Diagram_Batch_03_27-RandomForest.component.simple-narrative-l.ts=1724444716472.png/content/adobe-cms/us/en/topics/random-forest/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/simple_narrative0/image)

**[<u>Code Block</u>](https://www.kaggle.com/code/dansbecker/random-forests) :**

```python
import pandas as pd

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

```

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

```

### [<u>Preparing Data for Random Forest Modeling</u>](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/)

For Random Forest modeling, some key-steps of data preparation are discussed below:

- **Handling Missing Values:** Begin by addressing any missing values in the dataset. Techniques like imputation or removal of instances with missing values ensure a complete and reliable input for Random Forest.
- **Encoding Categorical Variables:** Random Forest requires numerical inputs, so categorical variables need to be encoded. Techniques like one-hot encoding or label encoding transform categorical features into a format suitable for the algorithm.
- **Scaling and Normalization:** While Random Forest is not sensitive to feature scaling, normalizing numerical features can still contribute to a more efficient training process and improved convergence.
- **Feature Selection:** Assess the importance of features within the dataset. Random Forest inherently provides a feature importance score, aiding in the selection of relevant features for model training.
- **Addressing Imbalanced Data:** If dealing with imbalanced classes, implement techniques like adjusting class weights or employing resampling methods to ensure a balanced representation during training.

## Difference between RandomForestClassifier and  RandomForestRegressor ?

The primary difference between **RandomForestClassifier** and **RandomForestRegressor** lies in the type of problem they are designed to solve and the outputs they produce. Here’s a comparison of the two:

### 1. **Purpose:**

- **RandomForestClassifier:**
  - Used for **classification problems**, where the goal is to predict a **categorical** label or class.
  - Example: Predicting whether an email is spam or not spam (binary classification), or classifying different species of flowers (multi-class classification).
  
- **RandomForestRegressor:**
  - Used for **regression problems**, where the goal is to predict a **continuous** value or number.
  - Example: Predicting the price of a house based on its features, such as location, size, etc.

### 2. **Output:**

- **RandomForestClassifier:**
  - The output is a **class label** (e.g., 0 or 1 in binary classification) or a probability distribution over classes (i.e., the likelihood of belonging to each class).
  
- **RandomForestRegressor:**
  - The output is a **continuous numerical value** (e.g., a predicted price, temperature, or score).

### 3. **Decision Making:**

- **RandomForestClassifier:**
  - Makes decisions by averaging votes from all decision trees in the forest and selecting the class with the **majority vote**.
  
- **RandomForestRegressor:**
  - Makes decisions by averaging the **predicted values** from all trees to provide a final numerical prediction.

### 4. **Evaluation Metrics:**

- **RandomForestClassifier:**
  - Common evaluation metrics include **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC**.
  
- **RandomForestRegressor:**
  - Common evaluation metrics include **mean squared error (MSE)**, **mean absolute error (MAE)**, **R-squared**, and **root mean squared error (RMSE)**.

### Summary

- **RandomForestClassifier** is for predicting categorical outcomes (classes).
- **RandomForestRegressor** is for predicting continuous outcomes (numbers).

## geekforgeeks : [Implement Random Forest for Classification](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/)

```python

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic_data = pd.read_csv(url)

# Drop rows with missing target values
titanic_data = titanic_data.dropna(subset=['Survived'])

# Select relevant features and target variable
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

# Convert categorical variable 'Sex' to numerical using .loc
X.loc[:, 'Sex'] = X['Sex'].map({'female': 0, 'male': 1})

# Handle missing values in the 'Age' column using .loc
X.loc[:, 'Age'].fillna(X['Age'].median(), inplace=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:\n', classification_rep)
```
## OUTPUT:
```

Accuracy: 0.80

Classification Report:
               precision    recall  f1-score   support

           0       0.82      0.85      0.83       105
           1       0.77      0.73      0.75        74

    accuracy                           0.80       179
   macro avg       0.79      0.79      0.79       179
weighted avg       0.80      0.80      0.80       179

```

## geekforgeeks : [Implement Random Forest for Regression](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/)

```python
# Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
california_housing = fetch_california_housing()
california_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_data['MEDV'] = california_housing.target

# Select relevant features and target variable
X = california_data.drop('MEDV', axis=1)
y = california_data['MEDV']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f&quot;Mean Squared Error: {mse:.2f}&quot;)
print(f&quot;R-squared Score: {r2:.2f}&quot;)

```
## OUTPUT:
```
Mean Squared Error: 0.26
R-squared Score: 0.81

```

## Sources/Credits:

- <https://www.kaggle.com/code/dansbecker/random-forests>
- <https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,both%20classification%20and%20regression%20problems>.
- <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>
- <https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/>
