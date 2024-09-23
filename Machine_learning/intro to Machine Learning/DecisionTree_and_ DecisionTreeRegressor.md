
## Decision Tree Classification Algorithm

[<u>Read more</u>](https://www.javatpoint.com/machine-learning-decision-tree-classification-algorithm)

- Decision Tree is a Supervised learning technique that can be used for both **classification and Regression** problems, but mostly it is preferred for solving Classification problems.
- It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.
- In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.
- The decisions or the test are performed on the basis of features of the given dataset.
- It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions.
- It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.
- In order to build a tree, we use the CART algorithm, which stands for Classification and Regression Tree algorithm.
- A decision tree simply asks a question, and based on the answer (Yes/No), it further split the tree into subtrees.

**Below diagram explains the general structure of a decision tree:**

![decision tree](https://d2jdgazzki9vjm.cloudfront.net/tutorial/machine-learning/images/decision-tree-classification-algorithm.png)

**Example:** Suppose there is a candidate who has a job offer and wants to decide whether he should accept the offer or Not. So, to solve this problem, the decision tree starts with the root node (Salary attribute by ASM). The root node splits further into the next decision node (distance from the office) and one leaf node based on the corresponding labels. The next decision node further gets split into one decision node (Cab facility) and one leaf node. Finally, the decision node splits into two leaf nodes (Accepted offers and Declined offer). 
- **Consider the below diagram:**

![example](https://d2jdgazzki9vjm.cloudfront.net/tutorial/machine-learning/images/decision-tree-classification-algorithm2.png)

# What is DecisionTreeRegressor ?

**Decision Tree** is a decision-making tool that uses a flowchart-like tree structure or is a model of decisions and all of their possible results, including outcomes, input costs, and utility.

**Decision-tree algorithm** falls under the category of <mark>supervised learning algorithms.</mark>  It works for both continuous as well as categorical output variables.

The branches/edges represent the result of the node and the nodes have either:

1. **Conditions** --> [Decision Nodes]

2. **Result**  --> [End Nodes]


**Decision Tree Regression:**
*Decision tree regression observes features of an object and trains a model in the structure of a tree to predict data in the future to produce meaningful continuous output.* Continuous output means that the output/result is not discrete, i.e., it is not represented just by a discrete, known set of numbers or values.

- **Discrete output example:** A weather prediction model that predicts whether or not there’ll be rain on a particular day.

- **Continuous output example:** A profit prediction model that states the probable profit that can be generated from the sale of a product.
Here, continuous values are predicted with the help of a decision tree regression model.

## How to import DecisionsTreeRegressor?

- `from sklearn.tree import DecisionTreeRegressor`

**EXAMPLE :**
[<U>Read more on implementation </U>](https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/)

```python
# import the regressor 
from sklearn.tree import DecisionTreeRegressor 

# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 42) 

# fit the regressor with X and Y data 
regressor.fit(X, y) 



```

*Code Block*

```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```
[DecisionTreeRegress and its application]()
---

## Sources:

- <https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting>
- <https://www.javatpoint.com/machine-learning-decision-tree-classification-algorithm>
- <https://www.geeksforgeeks.org/decision-tree-introduction-example/>
- <https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/>
- <https://stackoverflow.com/questions/56150132/how-to-build-a-decision-tree-regressor-model>
