"""Domain enumerations for the Hybrid Recommendation System."""
from enum import Enum


class ContentType(str, Enum):
    """Supported content/item verticals (extensible across products, videos, songs, books)."""
    ELECTRONICS = "electronics"
    BOOKS = "books"
    VIDEOS = "videos"
    SONGS = "songs"
    FASHION = "fashion"
    HOME_FITNESS = "home_fitness"


class CandidateSelectionStrategy(str, Enum):
    """Three strategies for querying candidate users in Collaborative Filtering (Graph DB)."""
    RANDOM_SAMPLE = "random_sample"          # 1. Pick a few users from the cluster at random
    TOP_N_SIMILAR = "top_n_similar"          # 2. Pick top-N similar users to target user
    ALL_IF_SMALL = "all_if_small"            # 3. If cluster is small enough, pick all (fallback to top_n/random)


class RecommendationSource(str, Enum):
    """Identifies which ML approach produced a recommendation."""
    CONTENT_FILTERING = "content_filtering_exploitation"
    COLLABORATIVE_FILTERING = "collaborative_filtering_exploration"
    HYBRID_BLENDED = "hybrid_blended"


class EventType(str, Enum):
    """Kafka event types emitted by frontend/API to trigger async recommendation generation."""
    RECOMMENDATIONS_EXHAUSTED = "RECOMMENDATIONS_EXHAUSTED"
    PRODUCT_VIEWED = "PRODUCT_VIEWED"
    PRODUCT_PURCHASED = "PRODUCT_PURCHASED"
    USER_CLUSTER_UPDATED = "USER_CLUSTER_UPDATED"