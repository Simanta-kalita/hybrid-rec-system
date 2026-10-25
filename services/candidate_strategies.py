"""
Strategy & Factory Patterns for Collaborative Filtering Candidate User Selection on the Graph DB.

Implements the three candidate user discovery strategies from the system design specification:
  1. RandomSampleCandidateStrategy: Pick a few users from the cluster at random (maximizes exploration).
  2. TopNSimilarCandidateStrategy: Pick top-N most similar users to our target user.
  3. SmallClusterAllCandidateStrategy: If the cluster is small enough (<= threshold), pick all users in it.
"""
from abc import ABC, abstractmethod
import random
from typing import Callable, Dict, List, Set, Tuple
from domain.entities import UserClusterMembership
from domain.enums import CandidateSelectionStrategy


class ICandidateUserStrategy(ABC):
    """Strategy interface for selecting candidate users from a user's Graph DB clusters."""

    @abstractmethod
    def select_candidates(
        self,
        target_user_id: str,
        memberships: List[UserClusterMembership],
        cluster_to_users: Dict[str, Set[str]],
        similarity_lookup: Callable[[str, str, str], float],
        top_n: int = 5,
        small_cluster_threshold: int = 4,
    ) -> List[Tuple[str, float, str]]:
        """Returns list of (candidate_user_id, similarity_score, shared_aspect) sorted by similarity descending."""
        pass


class RandomSampleCandidateStrategy(ICandidateUserStrategy):
    """Strategy 1: Pick a few users from each shared cluster at random."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def select_candidates(
        self,
        target_user_id: str,
        memberships: List[UserClusterMembership],
        cluster_to_users: Dict[str, Set[str]],
        similarity_lookup: Callable[[str, str, str], float],
        top_n: int = 5,
        small_cluster_threshold: int = 4,
    ) -> List[Tuple[str, float, str]]:
        candidate_map: Dict[str, Tuple[float, str]] = {}
        for m in memberships:
            peers = sorted(u for u in cluster_to_users.get(m.cluster_id, set()) if u != target_user_id)
            if not peers:
                continue
            sample_k = min(top_n, len(peers))
            sampled = self._rng.sample(peers, sample_k)
            for cand_id in sampled:
                sim = similarity_lookup(target_user_id, cand_id, m.aspect)
                if cand_id not in candidate_map or sim > candidate_map[cand_id][0]:
                    candidate_map[cand_id] = (sim, m.aspect)

        results = [(cid, sim, aspect) for cid, (sim, aspect) in candidate_map.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]


class TopNSimilarCandidateStrategy(ICandidateUserStrategy):
    """Strategy 2: Pick top-N most similar users to our target user across shared clusters."""

    def select_candidates(
        self,
        target_user_id: str,
        memberships: List[UserClusterMembership],
        cluster_to_users: Dict[str, Set[str]],
        similarity_lookup: Callable[[str, str, str], float],
        top_n: int = 5,
        small_cluster_threshold: int = 4,
    ) -> List[Tuple[str, float, str]]:
        candidate_map: Dict[str, Tuple[float, str]] = {}
        for m in memberships:
            peers = [u for u in cluster_to_users.get(m.cluster_id, set()) if u != target_user_id]
            scored = [(cand_id, similarity_lookup(target_user_id, cand_id, m.aspect)) for cand_id in peers]
            scored.sort(key=lambda x: x[1], reverse=True)
            for cand_id, sim in scored[:top_n]:
                if cand_id not in candidate_map or sim > candidate_map[cand_id][0]:
                    candidate_map[cand_id] = (sim, m.aspect)

        results = [(cid, sim, aspect) for cid, (sim, aspect) in candidate_map.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]


class SmallClusterAllCandidateStrategy(ICandidateUserStrategy):
    """Strategy 3: If a cluster is small enough (<= threshold), pick all users; otherwise fallback to top-N."""

    def select_candidates(
        self,
        target_user_id: str,
        memberships: List[UserClusterMembership],
        cluster_to_users: Dict[str, Set[str]],
        similarity_lookup: Callable[[str, str, str], float],
        top_n: int = 5,
        small_cluster_threshold: int = 4,
    ) -> List[Tuple[str, float, str]]:
        candidate_map: Dict[str, Tuple[float, str]] = {}
        for m in memberships:
            peers = [u for u in cluster_to_users.get(m.cluster_id, set()) if u != target_user_id]
            if len(peers) <= small_cluster_threshold:
                selected = [(cand_id, similarity_lookup(target_user_id, cand_id, m.aspect)) for cand_id in peers]
            else:
                scored = [(cand_id, similarity_lookup(target_user_id, cand_id, m.aspect)) for cand_id in peers]
                scored.sort(key=lambda x: x[1], reverse=True)
                selected = scored[:top_n]

            for cand_id, sim in selected:
                if cand_id not in candidate_map or sim > candidate_map[cand_id][0]:
                    candidate_map[cand_id] = (sim, m.aspect)

        results = [(cid, sim, aspect) for cid, (sim, aspect) in candidate_map.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class CandidateStrategyFactory:
    """Factory returning the requested ICandidateUserStrategy implementation."""

    def __init__(self, random_seed: int = 42) -> None:
        self._strategies: Dict[CandidateSelectionStrategy, ICandidateUserStrategy] = {
            CandidateSelectionStrategy.RANDOM_SAMPLE: RandomSampleCandidateStrategy(seed=random_seed),
            CandidateSelectionStrategy.TOP_N_SIMILAR: TopNSimilarCandidateStrategy(),
            CandidateSelectionStrategy.ALL_IF_SMALL: SmallClusterAllCandidateStrategy(),
        }

    def get_strategy(self, strategy: CandidateSelectionStrategy) -> ICandidateUserStrategy:
        return self._strategies.get(strategy, self._strategies[CandidateSelectionStrategy.TOP_N_SIMILAR])
