import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import BisectingKMeans
import pandas as pd

x = [3, 4, 5, 6, 7, 6, 7, 8, 3, 2, 3, 2]
y = [7, 6, 5, 4, 3, 2, 2, 4, 3, 6, 5, 4]
df = pd.DataFrame({'x':x,'y':y})
kmeans = KMeans(n_clusters=3, random_state=42) # random state 42 ensures same initial centroid 
clusters=kmeans.fit_predict(df)
df['clusters']=clusters
plt.scatter(x, y, c=kmeans.labels_)
plt.title("K-Means Clustering")
plt.show()
print("cluster as per K-Means Clustering:")
print(df)

minibatchkmeans=MiniBatchKMeans(n_clusters=3)
clusters=minibatchkmeans.fit_predict(df)
df['clusters']=clusters
plt.scatter(x, y, c=minibatchkmeans.labels_)
plt.title("Mini Batch K-Means Clustering")
plt.show()
print("cluster as per Mini Batch K-Means Clustering:")
print(df)

dbscan = DBSCAN(eps=2, min_samples=5)
clusters=dbscan.fit_predict(df)
df['clusters']=clusters
plt.scatter(x,y,c=dbscan.labels_)
plt.title("DBSCAN Clustering")
plt.show()
print("cluster as per DBSCAN Clustering:")
print(df)

bisectingkmeans = BisectingKMeans(n_clusters=3)
clusters=bisectingkmeans.fit_predict(df)
df['clusters'] = clusters
plt.scatter(x, y, c=bisectingkmeans.labels_)
plt.title("Bisecting K-Means Clustering")
plt.show()
print("cluster as per Bisecting K-Means Clustering:")
print(df)
