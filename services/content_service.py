"""
ML Approach 1 Service: Content Filtering [Exploitation] Recommendation Service.

When a user transacts, watches, reads, or views a product page:
  1. Loads the trained K-Means `ProductClusterModel` from S3.
  2. Finds which cluster the target product matches (or computes cosine similarity to centroids for unseen items).
  3. Computes Cosine Similarity between the target product vector and candidate products in that cluster
     (plus adjacent high-similarity products if the cluster has fewer than top_k items).
  4. Returns ranked similar items to exploit the user's immediate interest.
"""
from typing import List, Optional, Set
import numpy as np
from domain.entities import ProductClusterModel, RecommendationItem
from domain.enums import RecommendationSource
from infrastructure.interfaces import IObjectStore, IRelationalRepository
from ml_pipelines.content_pipeline import ContentFilteringSparkPipeline
from ml_pipelines.vectorizers import TFIDFVectorizer, cosine_similarity


class ContentFilteringService:
    """Real-time Content Filtering service using K-Means cluster lookup and Cosine Similarity ranking."""

    def __init__(self, relational_db: IRelationalRepository, s3_store: IObjectStore) -> None:
        self._db = relational_db
        self._s3 = s3_store

    def _load_model(self) -> Optional[ProductClusterModel]:
        return self._s3.get_object(
            bucket=ContentFilteringSparkPipeline.MODEL_BUCKET,
            key=ContentFilteringSparkPipeline.MODEL_KEY,
        )

    def recommend_similar_to_product(
        self,
        product_id: str,
        top_k: int = 5,
        exclude_product_ids: Optional[Set[str]] = None,
    ) -> List[RecommendationItem]:
        """
        Given a product the user is viewing or purchasing, finds its K-Means cluster
        and recommends the most similar products using Cosine Similarity.
        """
        model = self._load_model()
        if model is None:
            return []

        target_product = self._db.get_product(product_id)
        if target_product is None:
            return []

        excluded = set(exclude_product_ids or set())
        excluded.add(product_id)
        excluded.update(model.filtered_out_products)

        # Obtain vector and cluster ID for the target product
        if product_id in model.product_vectors:
            target_vec = np.array(model.product_vectors[product_id], dtype=np.float64)
            cluster_id = model.product_to_cluster[product_id]
        else:
            # Cold-start product vectorization using saved vocabulary & IDF weights
            vectorizer = TFIDFVectorizer()
            vectorizer.load_state(model.vocabulary, model.idf_weights)
            target_vec = vectorizer.transform([target_product.to_text_corpus()])[0]
            centroids = np.array(model.centroids, dtype=np.float64)
            sims = [cosine_similarity(target_vec, c) for c in centroids]
            cluster_id = int(np.argmax(sims))

        # Primary candidate pool: items in the same K-Means cluster
        same_cluster_pids = [
            pid for pid in model.cluster_to_products.get(cluster_id, [])
            if pid not in excluded
        ]

        # Score candidates inside the matched cluster using Cosine Similarity
        scored_candidates: List[RecommendationItem] = []
        seen_pids: Set[str] = set()

        for cand_pid in same_cluster_pids:
            cand_prod = self._db.get_product(cand_pid)
            if cand_prod is None:
                continue
            cand_vec = np.array(model.product_vectors[cand_pid], dtype=np.float64)
            sim_score = cosine_similarity(target_vec, cand_vec)
            # Combine cosine similarity with subtle quality rating boost
            final_score = round(0.85 * sim_score + 0.15 * (cand_prod.avg_rating / 5.0), 4)
            scored_candidates.append(
                RecommendationItem(
                    product_id=cand_prod.product_id,
                    title=cand_prod.title,
                    content_type=cand_prod.content_type,
                    score=final_score,
                    source=RecommendationSource.CONTENT_FILTERING,
                    explanation=(
                        f"Exploitation: Matched content cluster #{cluster_id} "
                        f"with cosine similarity {sim_score:.2f} to '{target_product.title}'"
                    ),
                    cluster_id=f"content_cluster_{cluster_id}",
                )
            )
            seen_pids.add(cand_pid)

        # If same-cluster candidates < top_k, backfill from nearest neighbors across other clusters
        if len(scored_candidates) < top_k:
            for cand_pid, vec_list in model.product_vectors.items():
                if cand_pid in excluded or cand_pid in seen_pids:
                    continue
                cand_prod = self._db.get_product(cand_pid)
                if cand_prod is None or cand_prod.content_type != target_product.content_type:
                    continue
                cand_vec = np.array(vec_list, dtype=np.float64)
                sim_score = cosine_similarity(target_vec, cand_vec)
                final_score = round(0.80 * sim_score + 0.15 * (cand_prod.avg_rating / 5.0), 4)
                cand_cid = model.product_to_cluster.get(cand_pid, -1)
                scored_candidates.append(
                    RecommendationItem(
                        product_id=cand_prod.product_id,
                        title=cand_prod.title,
                        content_type=cand_prod.content_type,
                        score=final_score,
                        source=RecommendationSource.CONTENT_FILTERING,
                        explanation=(
                            f"Exploitation: Same category ({cand_prod.content_type.value}) "
                            f"with cosine similarity {sim_score:.2f} to '{target_product.title}'"
                        ),
                        cluster_id=f"content_cluster_{cand_cid}",
                    )
                )

        scored_candidates.sort(key=lambda item: item.score, reverse=True)
        return scored_candidates[:top_k]

    def recommend_for_user_history(self, user_id: str, top_k: int = 5) -> List[RecommendationItem]:
        """Generates exploitation recommendations based on the user's recent orders and browsing history."""
        orders = self._db.get_user_orders(user_id)
        browsing = self._db.get_user_browsing_events(user_id)
        purchased_ids = {o.product_id for o in orders}

        seed_pids: List[str] = [o.product_id for o in reversed(orders)]
        for b in reversed(browsing):
            if b.product_id not in seed_pids:
                seed_pids.append(b.product_id)

        if not seed_pids:
            return []

        aggregated: dict[str, RecommendationItem] = {}
        for seed_pid in seed_pids[:3]:
            recs = self.recommend_similar_to_product(
                product_id=seed_pid,
                top_k=top_k,
                exclude_product_ids=purchased_ids,
            )
            for item in recs:
                if item.product_id not in aggregated or item.score > aggregated[item.product_id].score:
                    aggregated[item.product_id] = item

        results = list(aggregated.values())
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
