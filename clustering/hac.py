"""
Hierarchical Clustering (HAC) in Python
"""
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
 
# Using a dendrogram to find optimal # clusters
# Ward method - minimize variance within each cluster
# observe dendrogram - find longest vertical line
# draw horizontal threshold level through this line
# - count intersecting lines to find number of clusters to use
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward' ))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distances')
plt.show()

# Applying HAC to dataset
from sklearn.cluster import AgglomerativeClustering
hac = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean',
                              linkage = 'ward')
y_hac = hac.fit_predict(X)

# Visualizing the HAC results - clusters
plt.scatter(X[y_hac == 0, 0], X[y_hac == 0, 1],
            s = 100, c ='red', label = 'Cluster 1')
plt.scatter(X[y_hac == 1, 0], X[y_hac == 1, 1],
            s = 100, c ='blue', label = 'Cluster 2')
plt.scatter(X[y_hac == 2, 0], X[y_hac == 2, 1],
            s = 100, c ='green', label = 'Cluster 3')
plt.scatter(X[y_hac == 3, 0], X[y_hac == 3, 1],
            s = 100, c ='cyan', label = 'Cluster 4')
plt.scatter(X[y_hac == 4, 0], X[y_hac == 4, 1],
            s = 100, c ='magenta', label = 'Cluster 5')

plt.title('Clusters of clients')
plt.xlabel('Actual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()