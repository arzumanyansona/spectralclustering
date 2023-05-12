import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans
from numpy.linalg import eig
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans


class MySpectralClustering():
    def __init__(self, n_clusters, eps):
        self.n_clusters = n_clusters
        self.eps = eps

    def fit(self, X):
        self.X = X

        affinity = self.affinity_matrix(X)
        degree = self.degree_matrix(affinity)
        laplacian = self.laplacian_matrix(affinity, degree)
        eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        features = eigenvectors[:, :self.n_clusters]
        norm = np.linalg.norm(features, axis=1)
        features = features / norm[:, np.newaxis]
        kmeans = KMeans(n_clusters=self.n_clusters)
        labels = kmeans.fit_predict(features)
        self.labels_ = labels

        return self.labels_

    def affinity_matrix(self, X):
        affinity = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                if (euclidean_distances(X[i].reshape(1, -1), X[j].reshape(1, -1)) < self.eps and euclidean_distances(
                        X[i].reshape(1, -1), X[j].reshape(1, -1)) != 0):
                    affinity[i][j] = 1

        affinity = np.maximum(affinity, affinity.T)
        return affinity

    def degree_matrix(self, affinity):
        degree = np.diag(affinity.sum(axis=1))

        return degree

    def laplacian_matrix(self, affinity, degree):

        laplacian = degree - affinity
        return laplacian



X_moons, y_moons = make_moons(n_samples=100, noise=0.08, random_state=0)
plt.figure(figsize=(15, 7))
plt.scatter(X_moons[:, 0], X_moons[:, 1])
plt.show()

model1 = MySpectralClustering(n_clusters=2, eps=0.4)
model1.fit(X_moons)
plt.scatter(X_moons[:,0],X_moons[:,1], c=model1.labels_, cmap='rainbow')
plt.show()
# centers = [[1, 1], [-1, -1], [1, -1]]
# X_blobs, y_blobs = make_blobs(n_samples=100, centers=centers, cluster_std=0.3,
#                                   random_state=0)
# plt.figure(figsize=(15, 7))
# plt.scatter(X_blobs[:, 0], X_blobs[:, 1])
# plt.show()
#
# model1 = MySpectralClustering(n_clusters=3, eps=1.5)
# model1.fit(X_blobs)
# plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=model1.labels_, cmap='rainbow')
#
# X_circles, y_circles = make_circles(n_samples=100, noise=0.02, factor=0.6, random_state=0)
# plt.figure(figsize=(15, 7))
# plt.scatter(X_circles[:, 0], X_circles[:, 1])
# plt.show()
#
# model1 = MySpectralClustering(n_clusters=2, eps=1)
# model1.fit(X_circles)
# plt.scatter(X_circles[:, 0], X_circles[:, 1], c=model1.labels_, cmap='rainbow')
