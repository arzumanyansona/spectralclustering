import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class TSNE():
    def __init__(self, n_components=2, n_iteration=1000, learning_rate=500, perplexity=30, momentum=0.6):
        self.n_iteration = n_iteration
        self.learning_rate = learning_rate
        self.perplexity = perplexity
        self.n_components = n_components
        self.momentum = momentum

    def fit_transform(self, X):
        np.random.seed(42)
        distance_X = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                distance_X[i, j] = np.linalg.norm(X[i] - X[j])
        p = np.exp(-distance_X ** 2 / (2 * self.perplexity ** 2))
        np.fill_diagonal(p, 0)
        p /= p.sum(axis=1)

        Y = np.random.randn(X.shape[0], self.n_components)
        distance_Y = np.zeros(len(Y))
        q = np.zeros((len(Y), len(Y)))
        for i in range(len(Y)):
            distance_Y[i] = np.linalg.norm(Y[i] - Y)
            q[i] = (1 + (distance_Y[i]) ** 2) ** (-1)
        np.fill_diagonal(q, 0)
        q /= q.sum()

        dC = np.zeros((X.shape[0], self.n_components))

        for j in range(0, len(p)):
            dC[j] = (4 * np.dot((p[j] - q[j]), (Y[j] - Y)))
            dC[j] *= ((1 + (distance_Y[j] ** 2)) ** (-1))

        Y += self.momentum * Y - self.learning_rate * dC
        return Y


digits = load_digits()
plt.matshow(digits.images[0])
y_digits = digits.target
X_digits = digits.data
standardized_data = StandardScaler().fit_transform(X_digits)
data_1000 = standardized_data[0:1000, :]
labels_1000 = y_digits[0:1000]
tsne = TSNE()
X_digits_tsne = tsne.fit_transform(data_1000)
f = plt.figure(figsize = (8,8))
ax = plt.subplot(aspect = "equal")
for i in range(10):
     plt.scatter(X_digits_tsne[labels_1000== i, 0],
                X_digits_tsne[labels_1000 == i, 1])
plt.show()