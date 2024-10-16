Here’s a **Unsupervised Machine Learning & Scikit-learn Cheatsheet** that covers the key techniques and functions used in clustering and dimensionality reduction—two of the most commonly applied unsupervised learning methods.

---

### **1. Importing Libraries**
```python
# Basic imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
```

---

### **2. Loading and Preparing Data**
```python
# Load dataset (example with Pandas)
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)  # Only features, no target

# Feature scaling (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### **3. Clustering Algorithms**
#### **K-Means Clustering**
```python
from sklearn.cluster import KMeans

# Initialize the model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model and predict cluster labels
y_kmeans = kmeans.fit_predict(X_scaled)

# Cluster centers
print(kmeans.cluster_centers_)

# Inertia (sum of squared distances to the nearest cluster center)
print(kmeans.inertia_)
```

#### **Finding Optimal Number of Clusters (Elbow Method)**
```python
inertia = []
for n in range(1, 10):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot inertia to find the 'elbow'
import matplotlib.pyplot as plt
plt.plot(range(1, 10), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

#### **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
```python
from sklearn.cluster import DBSCAN

# Initialize the model
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Fit the model and predict cluster labels
y_dbscan = dbscan.fit_predict(X_scaled)

# Core samples and noise points (-1 represents noise)
print(np.unique(y_dbscan))
```

#### **Agglomerative Clustering (Hierarchical Clustering)**
```python
from sklearn.cluster import AgglomerativeClustering

# Initialize the model
agg_clust = AgglomerativeClustering(n_clusters=3)

# Fit the model and predict cluster labels
y_agg = agg_clust.fit_predict(X_scaled)
```

---

### **4. Dimensionality Reduction**
#### **Principal Component Analysis (PCA)**
```python
from sklearn.decomposition import PCA

# Initialize PCA model
pca = PCA(n_components=2)

# Fit PCA and transform the data
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print(pca.explained_variance_ratio_)

# Visualize the reduced data
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA Dimensionality Reduction')
plt.show()
```

#### **t-SNE (t-distributed Stochastic Neighbor Embedding)**
```python
from sklearn.manifold import TSNE

# Initialize t-SNE
tsne = TSNE(n_components=2, random_state=42)

# Fit and transform the data
X_tsne = tsne.fit_transform(X_scaled)

# Visualize the reduced data
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Dimensionality Reduction')
plt.show()
```

---

### **5. Evaluating Clustering Performance**
#### **Silhouette Score**
```python
from sklearn.metrics import silhouette_score

# Calculate silhouette score (higher is better, range: [-1, 1])
sil_score = silhouette_score(X_scaled, y_kmeans)
print(f'Silhouette Score: {sil_score}')
```

#### **Dendrogram for Hierarchical Clustering**
```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Perform hierarchical clustering using 'ward' method
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distance')
plt.show()
```

---

### **6. Hyperparameter Tuning for Clustering**
#### **Tuning K-Means with Different K-values**
```python
# Loop to evaluate K-Means with different cluster numbers (Silhouette Score)
for n in range(2, 10):
    kmeans = KMeans(n_clusters=n, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, y_kmeans)
    print(f'Number of Clusters: {n}, Silhouette Score: {sil_score}')
```

#### **Tuning DBSCAN with Different Parameters**
```python
# Tuning DBSCAN with different eps and min_samples values
dbscan = DBSCAN(eps=0.3, min_samples=10)
y_dbscan = dbscan.fit_predict(X_scaled)

# Evaluate performance with silhouette score
sil_score = silhouette_score(X_scaled, y_dbscan)
print(f'Silhouette Score: {sil_score}')
```

---

### **7. Visualizing Clusters**
#### **Plotting K-Means Clusters**
```python
# Scatter plot for visualizing clusters (if 2D data is available)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis')

# Plot the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)

plt.title('K-Means Clusters with Centers')
plt.show()
```

#### **Visualizing Clusters in PCA-Reduced Space**
```python
# Perform PCA for visualization (if data is high-dimensional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Scatter plot for visualizing clusters in reduced space
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='plasma')

plt.title('Clusters in PCA-Reduced Space')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()
```

---

### **8. Saving and Loading Models**
```python
import joblib

# Save model to a file
joblib.dump(kmeans, 'kmeans_model.pkl')

# Load model from a file
loaded_model = joblib.load('kmeans_model.pkl')
```

---

### **9. Anomaly Detection (Unsupervised Learning)**
#### **Using Isolation Forest**
```python
from sklearn.ensemble import IsolationForest

# Initialize Isolation Forest
isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

# Fit model
isolation_forest.fit(X_scaled)

# Predict anomalies (-1 indicates anomaly, 1 indicates normal)
y_pred = isolation_forest.predict(X_scaled)

# Count of anomalies
np.unique(y_pred, return_counts=True)
```

---

This cheatsheet covers common unsupervised learning techniques such as **clustering** (K-Means, DBSCAN, Agglomerative), **dimensionality reduction** (PCA, t-SNE), **performance evaluation**, and **visualization** techniques used in unsupervised learning workflows. It’s structured to offer a quick reference for applying these methods using **scikit-learn** in Python.