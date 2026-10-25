"""Vectorization and Cosine Similarity engine in NumPy for Content and Collaborative Filtering."""
import math
import re
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "your", "you", "or",
}


def tokenize(text: str) -> List[str]:
    """Extracts normalized alphanumeric tokens excluding stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors in n-dimensional space."""
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def cosine_similarity_matrix(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Computes pairwise cosine similarity matrix between rows of mat_a (M, D) and mat_b (N, D)."""
    norms_a = np.linalg.norm(mat_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(mat_b, axis=1, keepdims=True)
    norms_a = np.where(norms_a < 1e-12, 1.0, norms_a)
    norms_b = np.where(norms_b < 1e-12, 1.0, norms_b)
    normed_a = mat_a / norms_a
    normed_b = mat_b / norms_b
    return np.dot(normed_a, normed_b.T)


class TFIDFVectorizer:
    """Converts product metadata (title, description, type, tags) into L2-normalized n-dimensional vectors."""

    def __init__(self, max_features: int = 256) -> None:
        self.max_features = max_features
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray = np.array([], dtype=np.float64)

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        tokenized_docs = [tokenize(doc) for doc in documents]
        n_docs = len(tokenized_docs)
        doc_freq: Counter = Counter()
        term_freq_total: Counter = Counter()

        for tokens in tokenized_docs:
            term_freq_total.update(tokens)
            for unique_token in set(tokens):
                doc_freq[unique_token] += 1

        # Select top max_features tokens by frequency
        most_common = [term for term, _ in term_freq_total.most_common(self.max_features)]
        most_common.sort()
        self.vocabulary_ = {term: idx for idx, term in enumerate(most_common)}

        idf_list: List[float] = []
        for term in most_common:
            df = doc_freq[term]
            idf = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            idf_list.append(idf)
        self.idf_ = np.array(idf_list, dtype=np.float64)

        return self._transform_tokenized(tokenized_docs)

    def transform(self, documents: List[str]) -> np.ndarray:
        tokenized_docs = [tokenize(doc) for doc in documents]
        return self._transform_tokenized(tokenized_docs)

    def _transform_tokenized(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        n_docs = len(tokenized_docs)
        n_vocab = len(self.vocabulary_)
        matrix = np.zeros((n_docs, n_vocab), dtype=np.float64)

        for row_idx, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            counts = Counter(tokens)
            for term, count in counts.items():
                col_idx = self.vocabulary_.get(term)
                if col_idx is not None:
                    tf = 1.0 + math.log(count)
                    matrix[row_idx, col_idx] = tf * self.idf_[col_idx]

        # L2 normalize rows for unit-sphere cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return matrix / norms

    def export_state(self) -> Tuple[Dict[str, int], List[float]]:
        return dict(self.vocabulary_), self.idf_.tolist()

    def load_state(self, vocabulary: Dict[str, int], idf_weights: List[float]) -> None:
        self.vocabulary_ = dict(vocabulary)
        self.idf_ = np.array(idf_weights, dtype=np.float64)


