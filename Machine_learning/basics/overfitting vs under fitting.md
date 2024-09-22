

## **Underfitting in Machine Learning**
- A statistical model or a machine learning algorithm is said to have underfitting **when a model is too simple to capture data complexities.**
-  In simple terms, an underfit model’s are inaccurate, especially when applied to new, unseen examples. It mainly happens **when we uses very simple model with overly simplified assumptions**. To address underfitting problem of the model, we need to use more complex models, with enhanced feature representation, and less regularization.

**Reasons for Underfitting**
- The model is too simple, So it may be not capable to represent the complexities in the data.
- The input features which is used to train the model is not the adequate representations of underlying factors influencing the target variable.
- The size of the training dataset used is not enough.
- Excessive regularization are used to prevent the overfitting, which constraint the model to capture the data well.
- Features are not scaled.

**Techniques to Reduce Underfitting**
- Increase model complexity.
- Increase the number of features, performing feature engineering.
- Remove noise from the data.
- Increase the number of epochs or increase the duration of training to get better results.
---
## Overfitting in Machine Learning
A statistical model is said to be overfitted when the model does not make accurate predictions on testing data.When a model gets trained with **so much data, it starts learning from the noise and inaccurate data entries in our data set.** And when testing with test data results in High variance. (*basically  we have too much noise / complex unecessary data*)

**Reasons for Overfitting:**
- High variance and low bias.
- The model is too complex.
- The size of the training data.

**Techniques to Reduce Overfitting**
- Improving the quality of training data reduces overfitting by focusing on meaningful patterns, mitigate the risk of fitting the noise or irrelevant features.
- Increase the training data can improve the model’s ability to generalize to unseen data and reduce the likelihood of overfitting.
- Reduce model complexity.
- Early stopping during the training phase (have an eye over the loss over the training period as soon as loss begins to increase stop training).
- Ridge Regularization and Lasso Regularization.
- Use dropout for neural networks to tackle overfitting.

---

**Good Fit in a Statistical Model**
- Ideally, the case when the model makes the predictions with 0 error, is said to have a good fit on the data. This situation is achievable at a spot between overfitting and underfitting. In order to understand it, we will have to look at the performance of our model with the passage of time, while it is learning from the training dataset.

---

**Here's the takeaway: Models can suffer from either:**

- Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
- Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.

We use validation data, which isn't used in model training, to measure a candidate model's accuracy. This lets us try many candidate models and keep the best one.

## Reading Resources & sources/credits :

- [Underfitting and Overfitting Fine-tune your model for better performance. --> (Kaggle)](https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting)
- [underfitting-and-overfitting-in-machine-learning --> (geeksforgeeks)](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/)

## Get back repo? --> : [https://github.com/3XCeptional/Ml-and-Data-Science-Notes-Notebooks/](https://github.com/3XCeptional/Ml-and-Data-Science-Notes-Notebooks/)

