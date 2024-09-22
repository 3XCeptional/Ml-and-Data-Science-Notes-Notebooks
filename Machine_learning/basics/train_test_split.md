## [Model Validation](https://www.kaggle.com/code/dansbecker/model-validation?scriptVersionId=126670592&cellId=6)







```python
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 42) ## val == validation
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

```




## Get back repo? --> : [https://github.com/3XCeptional/Ml-and-Data-Science-Notes-Notebooks/](https://github.com/3XCeptional/Ml-and-Data-Science-Notes-Notebooks/)
