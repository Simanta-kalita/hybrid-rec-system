"""
ML Approach 2 Service: Collaborative Filtering [Exploration] Recommendation Engine on Graph DB.

Implements the exact 3-step graph discovery and link prediction workflow:
  1. Querying Candidate Users from Graph DB:
     - Strategy 1 (`RANDOM_SAMPLE`): Pick a few users from the cluster at random
     - Strategy 2 (`TOP_N_SIMILAR`): Pick top-N similar users to our user A
     - Strategy 3 (`ALL_IF_SMALL`): If cluster is small enough, pick all users
  2. Querying Candidate Items (Missing Edges in User-Product Bipartite Graph):
     - For each candidate user B, find items that B purchased (`[:PURCHASED]`) that user A did NOT purchase.
  3. Predicting & Ranking:
     - Predict how much user A would find each candidate item interesting (predicted rating / affinity score)
       as a weighted function of how similar user A and candidate user B are:
       predicted_rating(u, i) = sum(sim(u, v) * rating(v, i)) / sum(sim(u, v)) + multi_peer_support_bonus
     - Sort predictions in descending order and return top-N.
"""
from typing import List
from domain.entities import RecommendationItem
from domain.enums import CandidateSelectionStrategy, RecommendationSource
from infrastructure.interfaces import IGraphDatabase, IRelationalRepository


class CollaborativeFilteringService:
    """Executes Graph DB candidate user discovery, missing-edge item traversal, and rating prediction."""

    def __init__(self, graph_db: IGraphDatabase, relational_db: IRelationalRepository) -> None:
        self._graph = graph_db
        self._db = relational_db

    def recommend_exploratory_items(
        self,
        user_id: str,
        strategy: CandidateSelectionStrategy = CandidateSelectionStrategy.TOP_N_SIMILAR,
        top_n_users: int = 5,
        top_k_items: int = 5,
    ) -> List[RecommendationItem]:
        """
        Finds missing purchase edges in the Graph DB from similar cohort users and ranks them
        by predicted interest rating.
        """
        # Step 1: Query Candidate Users from User Clusters in Graph DB
        candidate_users = self._graph.query_candidate_users(
            user_id=user_id,
            strategy=strategy,
            top_n=top_n_users,
        )
        if not candidate_users:
            return []

        # Step 2: Query Candidate Items purchased by candidate users but NOT by target user
        candidate_items_map = self._graph.query_candidate_items(
            target_user_id=user_id,
            candidate_users=candidate_users,
        )
        if not candidate_items_map:
            return []

        # Step 3: Predict how much target user will find each candidate item interesting
        predictions: List[RecommendationItem] = []

        for product_id, endorsements in candidate_items_map.items():
            product = self._db.get_product(product_id)
            if product is None:
                continue

            # Weighted rating prediction: sum(sim(u, v) * r(v, i)) / sum(sim(u, v))
            weighted_sum = 0.0
            sim_total = 0.0
            max_sim = 0.0
            supporting_users: List[str] = []
            aspects_seen: List[str] = []

            for cand_uid, user_sim, rating, aspect in endorsements:
                effective_sim = max(0.05, user_sim)
                weighted_sum += effective_sim * rating
                sim_total += effective_sim
                if effective_sim > max_sim:
                    max_sim = effective_sim
                supporting_users.append(f"{cand_uid} (sim={effective_sim:.2f})")
                if aspect not in aspects_seen:
                    aspects_seen.append(aspect)

            predicted_rating = weighted_sum / sim_total if sim_total > 0 else 0.0
            # Normalize predicted interest into [0, 1] scale while preserving predicted rating in explanation
            # Combine predicted rating (out of 5.0), max peer similarity, and multi-user support bonus
            support_bonus = min(0.15, 0.05 * (len(endorsements) - 1))
            normalized_score = round(
                min(0.99, (predicted_rating / 5.0) * 0.65 + max_sim * 0.30 + support_bonus),
                4,
            )

            predictions.append(
                RecommendationItem(
                    product_id=product.product_id,
                    title=product.title,
                    content_type=product.content_type,
                    score=normalized_score,
                    source=RecommendationSource.COLLABORATIVE_FILTERING,
                    explanation=(
                        f"Exploration (Predicted Rating: {predicted_rating:.2f}/5.0): "
                        f"Purchased by similar users [{', '.join(supporting_users[:2])}] "
                        f"in cohort aspect '{', '.join(aspects_seen)}'"
                    ),
                    cluster_id=f"cohort_{aspects_seen[0]}" if aspects_seen else "cohort_global",
                )
            )

        # Arrange predictions in descending order and get top-N
        predictions.sort(key=lambda item: item.score, reverse=True)
        return predictions[:top_k_items]
