"""
Unified Hybrid Recommendation Facade Service.

Coordinates:
  - Serving precomputed recommendations from the KV Store (`user_id -> [product_1, product_2, ...]`).
  - Proactive Kafka event emission when a user consumes recommendations and the remaining unconsumed pool
    drops to or below `exhaustion_watermark` (or is completely exhausted).
  - Real-time product page recommendations via Content Filtering (Exploitation).
  - Direct cohort discovery via Collaborative Filtering (Exploration).
"""
from datetime import datetime, timezone
from typing import Any, Dict, List
import uuid
from domain.entities import BrowsingEvent, Order, RecommendationItem
from domain.enums import CandidateSelectionStrategy, EventType
from domain.events import KafkaEvent
from infrastructure.interfaces import (
    IEventBroker,
    IGraphDatabase,
    IRecommendationKVStore,
    IRelationalRepository,
)
from services.collaborative_service import CollaborativeFilteringService
from services.content_service import ContentFilteringService
from services.generator_worker import AsyncRecommendationGenerator


class HybridRecommendationFacade:
    """High-level application service orchestrating serving, consumption, and proactive Kafka triggers."""

    def __init__(
        self,
        relational_db: IRelationalRepository,
        graph_db: IGraphDatabase,
        kv_store: IRecommendationKVStore,
        kafka_broker: IEventBroker,
        content_service: ContentFilteringService,
        collaborative_service: CollaborativeFilteringService,
        generator_worker: AsyncRecommendationGenerator,
        exhaustion_watermark: int = 2,
    ) -> None:
        self._db = relational_db
        self._graph = graph_db
        self._kv = kv_store
        self._kafka = kafka_broker
        self._content_svc = content_service
        self._collab_svc = collaborative_service
        self._generator = generator_worker
        self.exhaustion_watermark = exhaustion_watermark

    def get_user_feed(self, user_id: str) -> Dict[str, Any]:
        """Fetches precomputed recommendations from KV Store; triggers Kafka generation if empty."""
        items = self._kv.get_recommendations(user_id=user_id, include_consumed=False)
        proactive_kafka_triggered = False

        if not items:
            # Feed is exhausted/empty -> push event to Kafka to regenerate
            event = KafkaEvent(
                event_type=EventType.RECOMMENDATIONS_EXHAUSTED,
                user_id=user_id,
                payload={"remaining": 0, "strategy": CandidateSelectionStrategy.TOP_N_SIMILAR.value},
                trigger_reason="KV store empty on feed fetch",
            )
            self._kafka.publish(event)
            proactive_kafka_triggered = True
            items = self._kv.get_recommendations(user_id=user_id, include_consumed=False)

        return {
            "user_id": user_id,
            "unconsumed_count": len(items),
            "proactive_kafka_triggered": proactive_kafka_triggered,
            "regeneration_cycles": self._generator.regeneration_count_by_user.get(user_id, 0),
            "recommendations": items,
        }

    def consume_user_feed(
        self,
        user_id: str,
        consume_count: int = 3,
        strategy: CandidateSelectionStrategy = CandidateSelectionStrategy.TOP_N_SIMILAR,
    ) -> Dict[str, Any]:
        """
        Simulates user scrolling/consuming items from their recommendation feed.
        When remaining unconsumed recommendations fall <= `exhaustion_watermark`,
        pushes a `RECOMMENDATIONS_EXHAUSTED` event to Kafka so the Recommendation Generator
        proactively refreshes the KV store!
        """
        consumed_items, remaining_before_trigger = self._kv.consume_recommendations(
            user_id=user_id,
            count=consume_count,
        )
        kafka_event_emitted = None

        if remaining_before_trigger <= self.exhaustion_watermark:
            kafka_event = KafkaEvent(
                event_type=EventType.RECOMMENDATIONS_EXHAUSTED,
                user_id=user_id,
                payload={
                    "remaining_before_trigger": remaining_before_trigger,
                    "watermark": self.exhaustion_watermark,
                    "strategy": strategy.value,
                },
                trigger_reason=(
                    f"Remaining recommendations ({remaining_before_trigger}) "
                    f"reached exhaustion watermark (<= {self.exhaustion_watermark})"
                ),
            )
            self._kafka.publish(kafka_event)
            kafka_event_emitted = kafka_event

        remaining_after = self._kv.get_remaining_count(user_id)
        return {
            "user_id": user_id,
            "consumed_items": consumed_items,
            "remaining_before_kafka_trigger": remaining_before_trigger,
            "proactive_kafka_triggered": kafka_event_emitted is not None,
            "kafka_event_id": kafka_event_emitted.event_id if kafka_event_emitted else None,
            "remaining_after_regeneration": remaining_after,
            "total_regenerations_for_user": self._generator.regeneration_count_by_user.get(user_id, 0),
        }

    def record_product_interaction(
        self,
        user_id: str,
        product_id: str,
        interaction_type: str = "purchase",
        rating: float = 5.0,
        dwell_time_seconds: float = 90.0,
    ) -> Dict[str, Any]:
        """
        Handles when a user transacts or views a product page:
          1. Persists the Order or BrowsingEvent to Relational DB & Graph DB.
          2. Immediately fetches Content Filtering (Exploitation) recommendations for that product page.
          3. Emits a Kafka event to asynchronously update the user's KV recommendation feed.
        """
        now = datetime.now(timezone.utc)
        if interaction_type == "purchase":
            order = Order(
                order_id=f"ord_{uuid.uuid4().hex[:8]}",
                user_id=user_id,
                product_id=product_id,
                implicit_rating=rating,
                timestamp=now,
            )
            self._db.add_order(order)
            self._graph.add_purchased_edge(
                user_id=user_id,
                product_id=product_id,
                rating=rating,
                timestamp=now.isoformat(),
            )
            event_type = EventType.PRODUCT_PURCHASED
        else:
            browsing = BrowsingEvent(
                event_id=f"brw_{uuid.uuid4().hex[:8]}",
                user_id=user_id,
                product_id=product_id,
                dwell_time_seconds=dwell_time_seconds,
                interaction_type=interaction_type,
                timestamp=now,
            )
            self._db.add_browsing_event(browsing)
            self._graph.add_viewed_edge(
                user_id=user_id,
                product_id=product_id,
                dwell_time=dwell_time_seconds,
            )
            event_type = EventType.PRODUCT_VIEWED

        # Fetch immediate product-page Content Filtering recommendations
        purchased_set = self._graph.get_user_purchased_products(user_id)
        immediate_similar_products = self._content_svc.recommend_similar_to_product(
            product_id=product_id,
            top_k=5,
            exclude_product_ids=purchased_set,
        )

        # Publish Kafka event to refresh user's async KV store feed
        kafka_event = KafkaEvent(
            event_type=event_type,
            user_id=user_id,
            payload={"product_id": product_id, "interaction_type": interaction_type},
            trigger_reason=f"User performed '{interaction_type}' on product '{product_id}'",
        )
        self._kafka.publish(kafka_event)

        return {
            "user_id": user_id,
            "product_id": product_id,
            "interaction_type": interaction_type,
            "kafka_event_id": kafka_event.event_id,
            "immediate_content_recommendations": immediate_similar_products,
        }
