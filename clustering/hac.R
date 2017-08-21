# Hierarchical Clustering (HAC) in R

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# Using the dendrogram to find the optimal # of clusters
dendrogram = hclust(dist(X, method = 'euclidean'),
                    method = 'ward.D')
plot(dendrogram,
     main = 'Dendrogram',
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Applying HAC to the dataset
hac = hclust(dist(X, method = 'euclidean'), 
             method = 'ward.D')
y_hac = cutree(hac, 5)

# Visualizing the clusters - for 2D clusters
library(cluster)
clusplot(X, y_hac, lines = 0 , shape = TRUE, 
         color = TRUE, labels = 2, plotchar = FALSE, 
         span = TRUE, main = paste('Clusters of clients'),
         xlab = 'Annual Income (k$)', ylab = 'Spending score (1-100)')