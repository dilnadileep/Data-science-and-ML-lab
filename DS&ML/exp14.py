from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_breast_cancer()
x = iris.data
y = iris.target

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x)
cluster_label=kmeans.labels_
print(cluster_label)

centroids=kmeans.cluster_centers_
print(centroids)

plt.scatter(x[:,0],x[:,1], c=cluster_label,cmap='viridis',marker='o',edgecolor='black')
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,c='red',label='centroid')

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("bleh2")
plt.legend()
plt.show()