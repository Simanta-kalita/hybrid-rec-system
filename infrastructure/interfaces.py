"""Abstract Base Classes (Interfaces) following SOLID / Dependency Inversion Principle for Low-Level Design."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from domain.entities import (
    BrowsingEvent,
    Order,
    Product,
    RecommendationItem,
    Review,
    User,
    UserClusterMembership,
)
from domain.enums import CandidateSelectionStrategy
from domain.events import KafkaEvent


class IRelationalRepository(ABC):
    """Interface for transactional databases (Products DB, Users DB, Orders DB, Browsing History DB)."""

    @abstractmethod
    def get_all_products(self) -> List[Product]:
        pass

    @abstractmethod
    def get_product(self, product_id: str) -> Optional[Product]:
        pass

    @abstractmethod
    def upsert_product(self, product: Product) -> None:
        pass

    @abstractmethod
    def get_all_users(self) -> List[User]:
        pass

    @abstractmethod
    def get_user(self, user_id: str) -> Optional[User]:
        pass

    @abstractmethod
    def upsert_user(self, user: User) -> None:
        pass

    @abstractmethod
    def get_all_orders(self) -> List[Order]:
        pass

    @abstractmethod
    def get_user_orders(self, user_id: str) -> List[Order]:
        pass

    @abstractmethod
    def add_order(self, order: Order) -> None:
        pass

    @abstractmethod
    def get_all_reviews(self) -> List[Review]:
        pass

    @abstractmethod
    def add_review(self, review: Review) -> None:
        pass

    @abstractmethod
    def get_all_browsing_events(self) -> List[BrowsingEvent]:
        pass

    @abstractmethod
    def get_user_browsing_events(self, user_id: str) -> List[BrowsingEvent]:
        pass

    @abstractmethod
    def add_browsing_event(self, event: BrowsingEvent) -> None:
        pass


class IObjectStore(ABC):
    """Interface for S3 Object Store (stores filtered ETL data lakes and MLlib K-Means models)."""

    @abstractmethod
    def put_object(self, bucket: str, key: str, data: Any) -> str:
        pass

    @abstractmethod
    def get_object(self, bucket: str, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def list_keys(self, bucket: str, prefix: str = "") -> List[str]:
        pass


class IGraphDatabase(ABC):
    """Interface for Neo4j-style Property Graph Database supporting User-Cluster-Product traversals."""

    @abstractmethod
    def upsert_user_node(self, user: User) -> None:
        pass

    @abstractmethod
    def upsert_product_node(self, product: Product) -> None:
        pass

    @abstractmethod
    def upsert_cluster_node(self, cluster_id: str, aspect: str, metadata: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def add_purchased_edge(self, user_id: str, product_id: str, rating: float, timestamp: str) -> None:
        pass

    @abstractmethod
    def add_viewed_edge(self, user_id: str, product_id: str, dwell_time: float) -> None:
        pass

    @abstractmethod
    def add_belongs_to_edge(self, membership: UserClusterMembership) -> None:
        pass

    @abstractmethod
    def add_similar_user_edge(self, user_a: str, user_b: str, aspect: str, similarity: float) -> None:
        pass

    @abstractmethod
    def get_user_clusters(self, user_id: str) -> List[UserClusterMembership]:
        pass

    @abstractmethod
    def query_candidate_users(
        self,
        user_id: str,
        strategy: CandidateSelectionStrategy = CandidateSelectionStrategy.TOP_N_SIMILAR,
        top_n: int = 5,
        small_cluster_threshold: int = 4,
    ) -> List[Tuple[str, float, str]]:
        """Returns list of (candidate_user_id, similarity_score, shared_aspect)."""
        pass

    @abstractmethod
    def query_candidate_items(
        self,
        target_user_id: str,
        candidate_users: List[Tuple[str, float, str]],
    ) -> Dict[str, List[Tuple[str, float, float, str]]]:
        """
        Finds missing edges: items purchased by candidate users that target_user has NOT purchased.
        Returns mapping: product_id -> list of (candidate_user_id, user_similarity, purchase_rating, aspect).
        """
        pass

    @abstractmethod
    def get_user_purchased_products(self, user_id: str) -> Set[str]:
        pass

    @abstractmethod
    def get_graph_summary(self) -> Dict[str, Any]:
        pass


class IRecommendationKVStore(ABC):
    """Interface for Recommendation KV Database { user_id -> [ product_1, product_2, ... ] }."""

    @abstractmethod
    def set_recommendations(self, user_id: str, items: List[RecommendationItem]) -> None:
        pass

    @abstractmethod
    def get_recommendations(self, user_id: str, include_consumed: bool = False) -> List[RecommendationItem]:
        pass

    @abstractmethod
    def consume_recommendations(self, user_id: str, count: int = 1) -> Tuple[List[RecommendationItem], int]:
        """Marks up to `count` unconsumed items as consumed and returns (consumed_items, remaining_unconsumed_count)."""
        pass

    @abstractmethod
    def get_remaining_count(self, user_id: str) -> int:
        pass


class IEventBroker(ABC):
    """Interface for Kafka event streaming bus supporting pub/sub for proactive recommendation generation."""

    @abstractmethod
    def publish(self, event: KafkaEvent) -> None:
        pass

    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[[KafkaEvent], None]) -> None:
        pass

    @abstractmethod
    def process_pending_events(self, topic: str = "recommendation-events") -> int:
        pass

    @abstractmethod
    def get_event_log(self, limit: int = 50) -> List[KafkaEvent]:
        pass
