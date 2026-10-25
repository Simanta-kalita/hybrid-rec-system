"""
ML Approach 2: Collaborative Filtering [Exploration] Pipeline (Spark + S3 + MLlib User Clustering + Neo4j Graph DB).
Inherits from `AbstractSparkClusteringJob` (Template Method Pattern) and uses `IProductQualityFilter`.
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set
import numpy as np
from domain.entities import Product, UserClusterMembership
from domain.filters import IProductQualityFilter, build_default_quality_filter_chain
from infrastructure.interfaces import (
    IGraphDatabase,
    IObjectStore,
    IRelationalRepository,
)
from ml_pipelines.base_spark_job import AbstractSparkClusteringJob
from ml_pipelines.kmeans_engine import SphericalKMeans
from ml_pipelines.vectorizers import cosine_similarity_matrix


class CollaborativeFilteringSparkPipeline(AbstractSparkClusteringJob):
    """Executes the Spark User Clustering & Graph DB Replication pipeline for Collaborative Filtering."""

    DATA_BUCKET = "rec-data-lake"
    USER_PROFILES_KEY = "collaborative/user_profiles.json"
    MODEL_BUCKET = "rec-models"
    USER_CLUSTERS_KEY = "collaborative/user_clusters.json"

    def __init__(
        self,
        relational_db: IRelationalRepository,
        s3_store: IObjectStore,
        graph_db: IGraphDatabase,
        min_rating_threshold: float = 3.2,
        max_return_rate: float = 0.30,
        quality_filter_chain: Optional[IProductQualityFilter] = None,
    ) -> None:
        self._db = relational_db
        self._s3 = s3_store
        self._graph = graph_db
        self.min_rating_threshold = min_rating_threshold
        self.max_return_rate = max_return_rate
        self._filter_chain = quality_filter_chain or build_default_quality_filter_chain(
            min_rating=min_rating_threshold,
            max_return_rate=max_return_rate,
        )

    def extract_from_databases(self) -> Dict[str, Any]:
        return {
            "users": self._db.get_all_users(),
            "products": self._db.get_all_products(),
            "orders": self._db.get_all_orders(),
            "browsing_events": self._db.get_all_browsing_events(),
        }

    def filter_and_transform(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        valid_products: Dict[str, Product] = {}
        for p in raw_data["products"]:
            is_valid, _ = self._filter_chain.evaluate(p)
            if is_valid:
                valid_products[p.product_id] = p

        if hasattr(self._graph, "clear_clusters_and_similarities"):
            self._graph.clear_clusters_and_similarities()

        for user in raw_data["users"]:
            self._graph.upsert_user_node(user)
        for product in valid_products.values():
            self._graph.upsert_product_node(product)

        user_item_weights: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for order in raw_data["orders"]:
            if order.product_id not in valid_products:
                continue
            self._graph.add_purchased_edge(
                user_id=order.user_id,
                product_id=order.product_id,
                rating=order.implicit_rating,
                timestamp=order.timestamp.isoformat(),
            )
            user_item_weights[order.user_id][order.product_id] += float(order.implicit_rating) * 1.5

        for event in raw_data["browsing_events"]:
            if event.product_id not in valid_products:
                continue
            self._graph.add_viewed_edge(
                user_id=event.user_id,
                product_id=event.product_id,
                dwell_time=event.dwell_time_seconds,
            )
            dwell_weight = min(2.5, 0.5 + (event.dwell_time_seconds / 60.0))
            user_item_weights[event.user_id][event.product_id] += dwell_weight

        return {
            "users": raw_data["users"],
            "orders_count": len(raw_data["orders"]),
            "browsing_count": len(raw_data["browsing_events"]),
            "valid_products": valid_products,
            "user_item_weights": user_item_weights,
        }

    def ingest_to_s3(self, filtered_data: Dict[str, Any]) -> Dict[str, str]:
        user_item_weights = filtered_data["user_item_weights"]
        s3_profiles_uri = self._s3.put_object(
            bucket=self.DATA_BUCKET,
            key=self.USER_PROFILES_KEY,
            data={uid: dict(weights) for uid, weights in user_item_weights.items()},
        )
        return {"s3_profiles_uri": s3_profiles_uri}

    def cluster_and_publish_artifacts(
        self,
        filtered_data: Dict[str, Any],
        s3_staging_uris: Dict[str, str],
    ) -> Dict[str, Any]:
        users = filtered_data["users"]
        valid_products: Dict[str, Product] = filtered_data["valid_products"]
        user_item_weights = filtered_data["user_item_weights"]

        aspect_to_pids: Dict[str, List[str]] = defaultdict(list)
        all_pids = sorted(valid_products.keys())
        for pid in all_pids:
            aspect_to_pids[valid_products[pid].content_type.value].append(pid)
        aspect_to_pids["global_behavior"] = all_pids

        user_ids = [u.user_id for u in users]
        all_memberships: List[UserClusterMembership] = []
        aspect_cluster_summary: Dict[str, int] = {}

        for aspect, pids in aspect_to_pids.items():
            if not pids:
                continue
            tag_set: Set[str] = set()
            for pid in pids:
                tag_set.update(valid_products[pid].tags)
            sorted_tags = sorted(tag_set)
            tag_to_idx = {t: i for i, t in enumerate(sorted_tags)}

            n_cols = len(pids) + len(sorted_tags)
            aspect_matrix = np.zeros((len(user_ids), n_cols), dtype=np.float64)
            active_user_indices: List[int] = []

            for u_idx, uid in enumerate(user_ids):
                weights = user_item_weights.get(uid, {})
                has_activity = False
                for p_col, pid in enumerate(pids):
                    w = weights.get(pid, 0.0)
                    if w > 0.0:
                        has_activity = True
                        aspect_matrix[u_idx, p_col] = w
                        for tag in valid_products[pid].tags:
                            t_col = len(pids) + tag_to_idx[tag]
                            aspect_matrix[u_idx, t_col] += w * 0.8
                if has_activity:
                    active_user_indices.append(u_idx)

            if len(active_user_indices) < 2:
                continue

            active_matrix = aspect_matrix[active_user_indices]
            n_clusters = min(2 if aspect != "global_behavior" else 3, len(active_user_indices))
            kmeans = SphericalKMeans(n_clusters=n_clusters, max_iter=50, random_state=42)
            kmeans.fit(active_matrix)

            for c_idx in range(n_clusters):
                cluster_id = f"cluster_{aspect}_{c_idx}"
                self._graph.upsert_cluster_node(
                    cluster_id=cluster_id,
                    aspect=aspect,
                    metadata={"centroid_dim": n_cols, "cluster_index": c_idx},
                )
            aspect_cluster_summary[aspect] = n_clusters

            multi_preds = kmeans.predict_multi_cluster(
                active_matrix,
                top_m=2 if aspect == "global_behavior" else 1,
                min_similarity=0.20,
            )

            for local_idx, cluster_list in enumerate(multi_preds):
                uid = user_ids[active_user_indices[local_idx]]
                for c_idx, sim_score in cluster_list:
                    cluster_id = f"cluster_{aspect}_{c_idx}"
                    membership = UserClusterMembership(
                        user_id=uid,
                        cluster_id=cluster_id,
                        aspect=aspect,
                        similarity_score=sim_score,
                    )
                    self._graph.add_belongs_to_edge(membership)
                    all_memberships.append(membership)

            pairwise_sim = cosine_similarity_matrix(active_matrix, active_matrix)
            for i_local, u_idx_a in enumerate(active_user_indices):
                uid_a = user_ids[u_idx_a]
                for j_local, u_idx_b in enumerate(active_user_indices):
                    if i_local == j_local:
                        continue
                    uid_b = user_ids[u_idx_b]
                    sim_val = float(pairwise_sim[i_local, j_local])
                    if sim_val > 0.05:
                        self._graph.add_similar_user_edge(
                            user_a=uid_a,
                            user_b=uid_b,
                            aspect=aspect,
                            similarity=sim_val,
                        )

        s3_clusters_uri = self._s3.put_object(
            bucket=self.MODEL_BUCKET,
            key=self.USER_CLUSTERS_KEY,
            data=[
                {
                    "user_id": m.user_id,
                    "cluster_id": m.cluster_id,
                    "aspect": m.aspect,
                    "similarity_score": m.similarity_score,
                }
                for m in all_memberships
            ],
        )

        return {
            "status": "SUCCESS",
            "users_processed": len(users),
            "orders_replicated_to_graph": filtered_data["orders_count"],
            "browsing_events_replicated": filtered_data["browsing_count"],
            "aspect_clusters_created": aspect_cluster_summary,
            "total_user_cluster_memberships": len(all_memberships),
            "s3_profiles_uri": s3_staging_uris["s3_profiles_uri"],
            "s3_clusters_uri": s3_clusters_uri,
        }
