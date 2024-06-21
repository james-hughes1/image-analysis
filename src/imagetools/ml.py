"""!@file ml.py
@brief Module containing implementations of machine learning algorithms.

@details Uses numpy to implement KMeans and Gradient Descent, in a way which
is not as optimised as, say, scikit-learn, but is readable.
@author Created by J. Hughes on 8th June 2024.
"""

import matplotlib.pyplot as plt
import numpy as np


class KMeans_Custom:
    """!@brief Implements a class that handles KMeans Clustering."""

    def __init__(self, K):
        self.K = K

    def assignment(self, data):
        """!@brief Assigns a matrix where each row represents the assigned
        centroid.

        @param data The data to assign labels to. Must be an (N, P) array.
        @return labels
        """
        N = data.shape[0]
        labels = np.zeros((N, self.K))
        for n in range(N):
            distances_n = np.sum((self.centroids - data[n, :]) ** 2, axis=1)
            labels[n, :] = np.arange(0, self.K) == np.argmin(distances_n)
            assert np.sum(labels[n, :]) == 1
        return labels

    def fit(self, data, verbose=10, limit=1e5):
        """!@brief Fits the KMeans clusters.

        @details Uses the KMeans++ algorithm to initialise centroids,
        and then iteratively updates them using Lloyd's algorithm.

        @param data The data to assign labels to. Must be an (N, P) array.
        @param verbose How often to print out number of reassigned labels.
        @param limit Maximum number of iterations before stopping. Prevents
        an indefinite loop.
        """
        rng = np.random.default_rng(seed=42)

        # Initialise problem variables
        N, P = data.shape

        # KMeans++ initialisation
        # ----------------------------------------------------------
        # D. Arthur and S. Vassilvitskii, “k-means++: the advantages of
        # careful seeding,” in Proceedings of the Eighteenth Annual ACM-SIAM
        # Symposium on Discrete Algorithms, ser. SODA ’07. USA: Society for
        # Industrial and Applied Mathematics, 2007, pp. 1027–1035
        # -----------------------------------------------------------

        centroids = np.zeros((self.K, P))
        # Choose first centroid randomly
        centroids[0, :] = data[rng.choice(list(range(N))), :]
        for k in range(1, self.K):
            # Find Nxk matrix of distances of data points to the k centroids
            # already chosen.
            distances_k = np.zeros((N, k))
            for j in range(k):
                distances_k[:, j] = np.sum((data - centroids[j]) ** 2, axis=1)
            # Probability of data point x_i being chosen is proportional to
            # the squared distance to the closest centroid already chosen,
            # hence favouring 'spread-out' centroids.
            prob = np.min(distances_k, axis=1)
            prob /= np.sum(prob)
            centroids[k, :] = data[rng.choice(list(range(N)), p=prob), :]

        self.centroids = centroids

        labels = self.assignment(data)

        # Lloyd's Algorithm
        # ----------------------------------------------------------
        # D. MacKay, “An Example Inference Task: Clustering” in Information
        # Theory, Inference and Learning Algorithms, Cambridge University
        # Press, 2003, ch. 20, pp. 284-292, Algorithm 20.2.
        # ----------------------------------------------------------

        converged = False
        iteration = 0
        # Loop until no more reassignments, or iteration limit reached.
        print("Commencing KMeans fitting...")
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
        """!@brief Returns an array whose rows are the closest centroid to
        each datum.

        @param data The data to assign labels to. Must be an (N, P) array.
        @return predictions (N, P) array of centroids
        """
        labels = self.assignment(data)
        return labels @ self.centroids

    def predict_cluster(self, data):
        """!@brief Returns an array whose entries are the closest centroid to
        each datum, mapped to a single identifying integer.

        @param data The data to assign labels to. Must be an (N, P) array.
        @return predictions (N, 1) array of predicted clusters
        """
        labels = self.assignment(data)
        return labels @ np.arange(self.K).reshape((-1, 1))


def gradient_descent(obj, grad, x0, obj_min, eps, lr, max_iters, filename):
    """!@brief Implements vanilla gradient descent for scalar functions on R2,
    where the true global minimum is known.

    @param obj The objective function to minimise.
    @param grad The gradient of the objective function (R^2-->R^2)
    @param x0 The initial iterate
    @param obj_min The minimal value of the objective function
    @param eps The threshold error for the stopping criterion
    @param lr The learning rate
    @param max_iters The maximum number of iterations to run before
    terminating
    @param filename Determines the name of the trajectory and loss plot
    file
    @return x estimated argmin (in R^2) of the objective
    """
    x = x0
    iteration = 0
    x_1_values = [x0[0]]
    x_2_values = [x0[1]]
    obj_values = [obj(x0)]
    # Loop until error less than or equal to eps., or max. iterations reached.
    while obj(x) - obj_min > eps and iteration < max_iters:
        iteration += 1
        x -= lr * grad(x)
        x_1_values.append(x[0])
        x_2_values.append(x[1])
        obj_values.append(obj(x))

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(x_1_values, x_2_values, marker="o")
    ax[1].plot(obj_values)
    ax[0].set(title="Trajectory")
    ax[1].set(
        title="Objective Function Value", xlabel="Iteration", ylabel="Value"
    )
    plt.tight_layout()
    plt.savefig(f"report/figures/{filename}.png")
    return x
