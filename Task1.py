
# 1. Import the required libraries.

# 2. Define the following 2-D (i.e. two feature) Data
# Subject A B
# 1 1.0 1.0
# 2 1.5 2.0
# 3 3.0 4.0
# 4 5.0 7.0
# 5 3.5 5.0
# 6 4.5 5.0
# 7 3.5 4.5

# 3. Fit k-means to the dataset, for k=2
# 4. Visualize
# 5. Observe effect of different values of k.
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd    
from sklearn.cluster import KMeans
a=np.array([1.0,1.5,3.0,5.0,3.5,4.5,3.5])
b=np.array([1.0,2.0,4.0,7.0,5.0,5.0,4.5])
x=np.stack((a,b),axis=0)
# wcss=[]

X=np.array([[1,1],[1.5,2],[3,4],[5,7],[3.5,5],[4.5,5],[3.5,4.5]])
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label ='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label ='Cluster 2')
#plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label ='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c ='yellow', label = 'Centroids')
plt.xlabel('A')
plt.ylabel('B')
plt.legend()
plt.show()
    
#     wcss_list.append(kmeans.inertia_)  
# mtp.plot(range(1, 11), wcss_list)  
# mtp.title('The Elobw Method Graph')  
# mtp.xlabel('Number of clusters(k)')  
# mtp.ylabel('wcss_list')  
# mtp.show()  

# print(x)