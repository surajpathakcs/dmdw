import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, BisectingKMeans

# Sample data
x = [3, 4, 5, 6, 7, 6, 7, 8, 3, 2, 3, 2]
y = [7, 6, 5, 4, 3, 2, 2, 4, 3, 6, 5, 4]
df = pd.DataFrame({"x": x, "y": y})

# ---- K-Means ----
kmeans = KMeans(n_clusters=3, random_state=1)
df["cluster_kmeans"] = kmeans.fit_predict(df)
plt.scatter(df.x, df.y, c=df.cluster_kmeans)
plt.title("K-Means Clustering")
plt.show()

# ---- Mini-Batch KMeans ----
mini_kmeans = MiniBatchKMeans(n_clusters=3, random_state=1)
df["cluster_mini"] = mini_kmeans.fit_predict(df)
plt.scatter(df.x, df.y, c=df.cluster_mini)
plt.title("Mini-Batch KMeans")
plt.show()

# ---- DBSCAN ----
dbscan = DBSCAN(eps=1.8, min_samples=3)
df["cluster_dbscan"] = dbscan.fit_predict(df)
plt.scatter(df.x, df.y, c=df.cluster_dbscan)
plt.title("DBSCAN Clustering")
plt.show()

# ---- Bisecting KMeans ----
bkm = BisectingKMeans(n_clusters=3, random_state=1)
df["cluster_bkm"] = bkm.fit_predict(df)
plt.scatter(df.x, df.y, c=df.cluster_bkm)
plt.title("Bisecting KMeans")
plt.show()

print("Cluster Results:")
print(df)
