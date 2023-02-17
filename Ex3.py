# Create your own dataset using the function make_blobs function from
# from the sklearn.datasets module and specifies the number of clusters
# (using parameter ‘centers’) as 7. Use elbow method to verify the clusters
# and visualize it. (optional exercise)
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X,y=make_blobs(n_samples=100,centers=7,n_features=2,random_state=0)
data=[]
# for i in range(0,lengt)
kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

option = {0 : 'red', 1 : 'pink', 2 : 'green', 3 : 'blue', 4 :'magenta', 5 : 'purple', 6 : 'orange', 7 : 'cyan'}
# Visualising the clusters
for i in range(0,7):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 100, c = option[i], label ="Cluster "+str(i) )
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label ='Cluster 2')
#plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label ='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c ='yellow', label = 'Centroids')
plt.xlabel('A')
plt.ylabel('B')
plt.legend()
plt.show()

#Elbow Method
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')