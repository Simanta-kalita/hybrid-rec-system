"""K-Means++ Clustering Engine (Spark MLlib equivalent) supporting both hard and multi-cluster soft assignments."""
from typing import List, Tuple
import numpy as np
from ml_pipelines.vectorizers import cosine_similarity_matrix


class SphericalKMeans:
    """
    K-Means++ clustering operating on L2-normalized feature vectors (Spherical K-Means).
    Maximizes cosine similarity between cluster members and cluster centroids.
    """

    def __init__(self, n_clusters: int = 4, max_iter: int = 50, tol: float = 1e-4, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.labels_: np.ndarray = np.empty((0,), dtype=np.int64)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return matrix / norms

    def _init_kmeans_plus_plus(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples, n_features = X.shape
        k = min(self.n_clusters, n_samples)
        centroids = np.zeros((k, n_features), dtype=np.float64)

        # Pick first centroid deterministically/randomly
        first_idx = int(rng.integers(0, n_samples))
        centroids[0] = X[first_idx]

        for c_id in range(1, k):
            sim_to_existing = cosine_similarity_matrix(X, centroids[:c_id])
            max_sim = np.max(sim_to_existing, axis=1)
            # Distance in cosine space = 1 - max_sim
            distances_sq = np.maximum(1e-8, (1.0 - max_sim) ** 2)
            probs = distances_sq / np.sum(distances_sq)
            next_idx = int(rng.choice(n_samples, p=probs))
            centroids[c_id] = X[next_idx]

        return self._normalize_rows(centroids)

    def fit(self, X: np.ndarray) -> "SphericalKMeans":
        X_norm = self._normalize_rows(X)
        n_samples, n_features = X_norm.shape
        k = min(self.n_clusters, max(1, n_samples))
        rng = np.random.default_rng(self.random_state)

        centroids = self._init_kmeans_plus_plus(X_norm, rng)
        labels = np.zeros(n_samples, dtype=np.int64)

        for _ in range(self.max_iter):
            # Assign each point to the centroid with highest cosine similarity
            sim_matrix = cosine_similarity_matrix(X_norm, centroids)
            new_labels = np.argmax(sim_matrix, axis=1)

            new_centroids = np.zeros_like(centroids)
            for cluster_idx in range(k):
                members = X_norm[new_labels == cluster_idx]
                if len(members) > 0:
                    new_centroids[cluster_idx] = np.mean(members, axis=0)
                else:
                    # Re-initialize empty cluster with farthest sample
                    farthest_idx = int(np.argmin(np.max(sim_matrix, axis=1)))
                    new_centroids[cluster_idx] = X_norm[farthest_idx]

            new_centroids = self._normalize_rows(new_centroids)
            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            labels = new_labels
            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (best_cluster_indices, best_cosine_similarities)."""
        X_norm = self._normalize_rows(X)
        sim_matrix = cosine_similarity_matrix(X_norm, self.centroids_)
        best_clusters = np.argmax(sim_matrix, axis=1)
        best_sims = np.max(sim_matrix, axis=1)
        return best_clusters, best_sims

    def predict_multi_cluster(
        self,
        X: np.ndarray,
        top_m: int = 2,
        min_similarity: float = 0.15,
    ) -> List[List[Tuple[int, float]]]:
        """
        Supports multi-cluster membership: a user can belong to multiple clusters if their similarity
        exceeds `min_similarity`. Always assigns at least the top-1 cluster if the user has non-zero activity.
        """
        X_norm = self._normalize_rows(X)
        sim_matrix = cosine_similarity_matrix(X_norm, self.centroids_)
        multi_assignments: List[List[Tuple[int, float]]] = []

        for row_idx in range(X_norm.shape[0]):
            sims = sim_matrix[row_idx]
            ranked_indices = np.argsort(sims)[::-1]
            row_clusters: List[Tuple[int, float]] = []
            for rank, c_idx in enumerate(ranked_indices[:top_m]):
                score = float(sims[c_idx])
                if score >= min_similarity or (rank == 0 and score > 0.0):
                    row_clusters.append((int(c_idx), round(score, 4)))
            multi_assignments.append(row_clusters)

        return multi_assignments