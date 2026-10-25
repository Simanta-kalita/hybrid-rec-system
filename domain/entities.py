"""Domain entities representing Products, Users, Orders, Reviews, Browsing History, and Recommendations."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from domain.enums import ContentType, RecommendationSource


@dataclass
class Product:
    """Represents a catalog item (product, book, video, song, creator)."""
    product_id: str
    title: str
    description: str
    content_type: ContentType
    tags: List[str]
    price: float
    avg_rating: float = 0.0
    review_count: int = 0
    return_rate: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_text_corpus(self) -> str:
        """Combines title, description, type, and tags for TF-IDF vectorization."""
        tag_str = " ".join(self.tags * 2)  # Boost tag weight
        type_str = f"{self.content_type.value} {self.content_type.value}"
        return f"{self.title} {self.description} {type_str} {tag_str}".lower()


@dataclass
class Review:
    """User product review used by the Spark ETL job to filter out bad products."""
    review_id: str
    user_id: str
    product_id: str
    rating: float  # 1.0 to 5.0
    comment: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class User:
    """Platform user entity."""
    user_id: str
    name: str
    preferred_categories: List[ContentType] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Order:
    """Purchase record connecting a User and a Product (replicated to Graph DB)."""
    order_id: str
    user_id: str
    product_id: str
    quantity: int = 1
    implicit_rating: float = 5.0  # Explicit review rating or implicit purchase rating
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BrowsingEvent:
    """User browsing/consumption history used alongside orders for user clustering."""
    event_id: str
    user_id: str
    product_id: str
    dwell_time_seconds: float
    interaction_type: str = "view"  # view, watch, read, listen, search_click
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProductClusterModel:
    """Artifact stored in S3 representing trained K-Means product clusters."""
    model_version: str
    num_clusters: int
    vocabulary: Dict[str, int]
    idf_weights: List[float]
    centroids: List[List[float]]  # Shape: (k, n_features)
    product_to_cluster: Dict[str, int]
    cluster_to_products: Dict[int, List[str]]
    product_vectors: Dict[str, List[float]]
    filtered_out_products: List[str]
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UserClusterMembership:
    """Represents a user's membership in an aspect/cohort cluster (multi-cluster support)."""
    user_id: str
    cluster_id: str
    aspect: str  # e.g., "books", "electronics", "global_behavior"
    similarity_score: float  # Cosine similarity to cluster centroid


@dataclass
class RecommendationItem:
    """A single scored recommendation item stored in the KV Store and served to clients."""
    product_id: str
    title: str
    content_type: ContentType
    score: float
    source: RecommendationSource
    explanation: str
    cluster_id: Optional[str] = None
    consumed: bool = False
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


