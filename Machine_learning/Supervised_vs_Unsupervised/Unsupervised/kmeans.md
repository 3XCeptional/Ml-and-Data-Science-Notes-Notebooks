
# K-Means Cluster:

1. K-means is an unsupervised learning method for clustering data points. The algorithm iteratively divides data points into K clusters by minimizing the variance in each cluster.

2. K-Means clustering is the most popular unsupervised learning algorithm. It is used when we have unlabelled data which is data without defined categories or groups. The algorithm follows an easy or simple way to classify a given data set through a certain number of clusters, fixed apriori. K-Means algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.

![kmeans](https://miro.medium.com/max/2160/1*tWaaZX75oumVwBMcKN-eHA.png)

Read more : [Introduction to K-Means Clustering ](https://www.kaggle.com/code/prashant111/k-means-clustering-with-python)

### how to use K-means
- Import statement `from sklearn.cluster import KMeans` (note : pip install scikit-learn)



```py

import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12,13]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#Two  lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()


```

## What is Elbow Method ? -->[source](https://builtin.com/data-science/elbow-method#:~:text=The%20elbow%20method%20is%20a%20graphical%20method%20for%20finding%20the,on%20the%20x%2Daxis)

- The elbow method is a graphical representation of finding the optimal ‘K’ in a k-means clustering. This is typically done by picking out the k-value where the elbow is created. However, this is not the best way to find the optimal ‘K’.   

## The Following Diagrams shows elbow method :
![elbow method](images/elbow.png)
- The elbow method shows that 2 is a good value for K, so we retrain and visualize the result:
- We can then fit our K-means algorithm one more time and plot the different clusters assigned to the data:

- [More Resources](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/)

Sample Code :
```py
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()
```
### result :  2 Cluster
![Result](images/result.png)

## Sample kmeans code

```py
k_means3 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.show()
```
### Checkout my Notebooks  on K-Means Clustering :

- [Notebooks : Kmeans](/DataScience_and_ML_Notebooks/kmeans)


## More Resources :
- [K-Means Clustering with Python](https://www.kaggle.com/code/prashant111/k-means-clustering-with-python)
- [Source : www.w3schools](https://www.w3schools.com/python/python_ml_k-means.asp)
- [geekforgeeks](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/)