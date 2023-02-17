# There is a big mall in a specific city that keeps information of its customers
# who subscribe to a membership card. In the membership card they provide
# following information: gender, age and annual income. The customers use this
# membership card to make all the purchases in the mall, so that mall has the
# purchase history of all subscribed members and according to that they
# compute the Spending score of all customers. Segment these customers using
# k-means clustering based on the details given. Download the dataset from:
# https://github.com/stavanR/Machine-Learning-Algorithms-Dataset.
# Dataset file name: Mall_Customers.csv
# 1. Import the required libraries
# 2. Import the Dataset
# 3. Find optimal number of clusters
# 4. Fit k-means to the dataset
# 5. Visualize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
dataset=pd.read_csv('/media/ubantu/New Volume/Local disk/6th Sem/Machine Learning/Exp 2/Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label ='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label ='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label ='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label ='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label ='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c ='yellow', label = 'Centroids')
plt.title('Cluster of Customers')
plt.xlabel('Annual income')
plt.ylabel('Spending Scores')
plt.legend()
plt.show()