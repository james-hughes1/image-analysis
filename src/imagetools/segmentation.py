import numpy as np


class KMeans_Custom:
    def __init__(self, K):
        self.K = K

    def assignment(self, data):
        N = data.shape[0]
        labels = np.zeros((N, self.K))
        for n in range(N):
            distances_n = np.sum((self.centroids - data[n, :]) ** 2, axis=1)
            labels[n, :] = np.arange(0, self.K) == np.argmin(distances_n)
            assert np.sum(labels[n, :]) == 1
        return labels

    def fit(self, data, verbose=10, limit=1e5):
        """Based on [reference]"""
        rng = np.random.default_rng(seed=42)

        # Initialisation
        N, P = data.shape

        # kmeans++
        centroids = np.zeros((self.K, P))
        centroids[0, :] = data[rng.choice(list(range(N))), :]
        for k in range(1, self.K):
            distances_k = np.zeros((N, k))
            for j in range(k):
                distances_k[:, j] = np.sum((data - centroids[j]) ** 2, axis=1)
            prob = np.min(distances_k, axis=1)
            prob /= np.sum(prob)
            centroids[k, :] = data[rng.choice(list(range(N)), p=prob), :]

        self.centroids = centroids

        labels = self.assignment(data)

        converged = False

        iteration = 0
        while not converged and iteration < limit:
            iteration += 1
            if not iteration % verbose:
                print(f"Iteration {iteration}")
            # Update step
            centroids = (
                labels.T / (np.sum(labels.T, axis=1, keepdims=True))
            ) @ data
            self.centroids = centroids

            labels_new = self.assignment(data)
            reassigned = int(np.abs(labels_new - labels).sum()) // 2
            # Assignment step
            if reassigned == 0:
                converged = True
            if not iteration % verbose:
                print(f"No. re-assigned={reassigned}")
            labels = self.assignment(data)
        self.centroids = centroids

    def predict(self, data):
        labels = self.assignment(data)
        return labels @ self.centroids

    def predict_cluster(self, data):
        labels = self.assignment(data)
        return labels @ np.arange(self.K).reshape((-1, 1))
