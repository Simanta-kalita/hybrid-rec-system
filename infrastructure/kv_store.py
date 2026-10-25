"""Simulated Redis Key-Value Store holding precomputed recommendations: { user_id -> [product_1, product_2, ...] }."""
import copy
from typing import Dict, List, Tuple
from domain.entities import RecommendationItem
from infrastructure.interfaces import IRecommendationKVStore


class RedisRecommendationKVStore(IRecommendationKVStore):
    """
    Fast O(1) Key-Value Store for serving user recommendations.
    Tracks unconsumed vs. consumed items so the API can detect when a user's feed is about to exhaust
    and proactively emit a Kafka event to trigger async regeneration.
    """

    def __init__(self) -> None:
        self._store: Dict[str, List[RecommendationItem]] = {}

    def set_recommendations(self, user_id: str, items: List[RecommendationItem]) -> None:
        self._store[user_id] = copy.deepcopy(items)

    def get_recommendations(self, user_id: str, include_consumed: bool = False) -> List[RecommendationItem]:
        items = self._store.get(user_id, [])
        if include_consumed:
            return copy.deepcopy(items)
        return [copy.deepcopy(item) for item in items if not item.consumed]

    def consume_recommendations(self, user_id: str, count: int = 1) -> Tuple[List[RecommendationItem], int]:
        items = self._store.get(user_id, [])
        consumed_batch: List[RecommendationItem] = []
        for item in items:
            if not item.consumed and len(consumed_batch) < count:
                item.consumed = True
                consumed_batch.append(copy.deepcopy(item))

        remaining = sum(1 for item in items if not item.consumed)
        return consumed_batch, remaining

    def get_remaining_count(self, user_id: str) -> int:
        items = self._store.get(user_id, [])
        return sum(1 for item in items if not item.consumed)