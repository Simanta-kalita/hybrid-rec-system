"""
Async Recommendation Generator Worker (Kafka Consumer + Batch Scoring & Ranking Job).

Matches Whiteboard Diagrams 1 & 2:
  - Runs on top of Graph DB (Collaborative Filtering - Exploration) + Content Filtering Model (Exploitation).
  - Generates a ranked feed of recommendations for each user and writes them to the Recommendation KV Database:
      { user_id -> [ product_1, product_2, ... ] }
  - Subscribes to the `recommendation-events` Kafka topic so whenever a user's recommendations are about to
    "exhaust" (or when a user transacts/views an item), it asynchronously regenerates fresh recommendations.
"""
from typing import Dict, List
from domain.entities import RecommendationItem
from domain.enums import CandidateSelectionStrategy, RecommendationSource
from domain.events import KafkaEvent
from infrastructure.interfaces import (
    IEventBroker,
    IRecommendationKVStore,
    IRelationalRepository,
)
from services.collaborative_service import CollaborativeFilteringService
from services.content_service import ContentFilteringService


class AsyncRecommendationGenerator:
    """
    Asynchronous worker that blends Content Filtering (Exploitation) and Collaborative Filtering (Exploration)
    and populates the Recommendation KV Store (`user_id -> [products]`).
    """

    def __init__(
        self,
        relational_db: IRelationalRepository,
        kv_store: IRecommendationKVStore,
        content_service: ContentFilteringService,
        collaborative_service: CollaborativeFilteringService,
        kafka_broker: IEventBroker,
        batch_size: int = 8,
    ) -> None:
        self._db = relational_db
        self._kv = kv_store
        self._content_svc = content_service
        self._collab_svc = collaborative_service
        self._kafka = kafka_broker
        self.batch_size = batch_size
        self.regeneration_count_by_user: Dict[str, int] = {}

        # Subscribe this generator worker to Kafka topic
        self._kafka.subscribe("recommendation-events", self.handle_kafka_event)

    def handle_kafka_event(self, event: KafkaEvent) -> None:
        """Kafka consumer callback triggered when frontend/API emits an exhaustion or interaction event."""
        strategy_name = event.payload.get("strategy", CandidateSelectionStrategy.TOP_N_SIMILAR.value)
        try:
            strategy = CandidateSelectionStrategy(strategy_name)
        except ValueError:
            strategy = CandidateSelectionStrategy.TOP_N_SIMILAR

        self.generate_for_user(
            user_id=event.user_id,
            strategy=strategy,
            exploitation_ratio=float(event.payload.get("exploitation_ratio", 0.5)),
        )

    def generate_for_user(
        self,
        user_id: str,
        strategy: CandidateSelectionStrategy = CandidateSelectionStrategy.TOP_N_SIMILAR,
        exploitation_ratio: float = 0.5,
    ) -> List[RecommendationItem]:
        """
        Generates a hybrid list of recommendations combining:
          - Content Filtering (Exploitation: give people more of what they consume)
          - Collaborative Filtering (Exploration: spice up recommendations with what similar users bought)
        Stores the resulting list in the Recommendation KV Store under `user_id`.
        """
        exploit_quota = max(1, int(round(self.batch_size * exploitation_ratio)))
        explore_quota = max(1, self.batch_size - exploit_quota)

        content_recs = self._content_svc.recommend_for_user_history(
            user_id=user_id,
            top_k=exploit_quota + 3,
        )
        collab_recs = self._collab_svc.recommend_exploratory_items(
            user_id=user_id,
            strategy=strategy,
            top_n_users=5,
            top_k_items=explore_quota + 3,
        )

        # Interleave and deduplicate items (if an item appears in both, mark as HYBRID_BLENDED and boost score)
        content_map = {item.product_id: item for item in content_recs}
        collab_map = {item.product_id: item for item in collab_recs}

        blended_list: List[RecommendationItem] = []
        used_pids: set[str] = set()

        # First promote items endorsed by BOTH Content Filtering and Collaborative Filtering
        for pid, c_item in content_map.items():
            if pid in collab_map:
                cf_item = collab_map[pid]
                hybrid_score = round(min(0.99, 0.5 * c_item.score + 0.5 * cf_item.score + 0.08), 4)
                blended_list.append(
                    RecommendationItem(
                        product_id=pid,
                        title=c_item.title,
                        content_type=c_item.content_type,
                        score=hybrid_score,
                        source=RecommendationSource.HYBRID_BLENDED,
                        explanation=f"{c_item.explanation} | PLUS {cf_item.explanation}",
                        cluster_id=f"{c_item.cluster_id}+{cf_item.cluster_id}",
                    )
                )
                used_pids.add(pid)

        # Interleave remaining Exploitation and Exploration recommendations
        c_idx, cf_idx = 0, 0
        while len(blended_list) < self.batch_size and (c_idx < len(content_recs) or cf_idx < len(collab_recs)):
            if c_idx < len(content_recs):
                cand = content_recs[c_idx]
                c_idx += 1
                if cand.product_id not in used_pids:
                    blended_list.append(cand)
                    used_pids.add(cand.product_id)
            if len(blended_list) >= self.batch_size:
                break
            if cf_idx < len(collab_recs):
                cand = collab_recs[cf_idx]
                cf_idx += 1
                if cand.product_id not in used_pids:
                    blended_list.append(cand)
                    used_pids.add(cand.product_id)

        blended_list.sort(key=lambda item: item.score, reverse=True)
        self._kv.set_recommendations(user_id=user_id, items=blended_list)
        self.regeneration_count_by_user[user_id] = self.regeneration_count_by_user.get(user_id, 0) + 1
        return blended_list

    def generate_for_all_users(self) -> Dict[str, int]:
        """Batch job running on top of Graph DB + Content Model to pre-populate the KV Store for all users."""
        users = self._db.get_all_users()
        counts: Dict[str, int] = {}
        for user in users:
            items = self.generate_for_user(user.user_id)
            counts[user.user_id] = len(items)
        return counts
