"""
ML Approach 1: Content Filtering [Exploitation] Pipeline (Spark + S3 + MLlib K-Means).
Inherits from `AbstractSparkClusteringJob` (Template Method) and uses `IProductQualityFilter` (Chain of Responsibility).
"""
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
from domain.entities import Product, ProductClusterModel
from domain.filters import IProductQualityFilter, build_default_quality_filter_chain
from infrastructure.interfaces import IObjectStore, IRelationalRepository
from ml_pipelines.base_spark_job import AbstractSparkClusteringJob
from ml_pipelines.kmeans_engine import SphericalKMeans
from ml_pipelines.vectorizers import TFIDFVectorizer


class ContentFilteringSparkPipeline(AbstractSparkClusteringJob):
    """Executes the batch Spark ETL + MLlib K-Means clustering pipeline for Content Filtering."""

    DATA_BUCKET = "rec-data-lake"
    FILTERED_PRODUCTS_KEY = "content/filtered_products.json"
    MODEL_BUCKET = "rec-models"
    MODEL_KEY = "content/kmeans_product_model.json"

    def __init__(
        self,
        relational_db: IRelationalRepository,
        s3_store: IObjectStore,
        num_clusters: int = 5,
        min_rating_threshold: float = 3.2,
        max_return_rate: float = 0.30,
        quality_filter_chain: Optional[IProductQualityFilter] = None,
    ) -> None:
        self._db = relational_db
        self._s3 = s3_store
        self.num_clusters = num_clusters
        self.min_rating_threshold = min_rating_threshold
        self.max_return_rate = max_return_rate
        self._filter_chain = quality_filter_chain or build_default_quality_filter_chain(
            min_rating=min_rating_threshold,
            max_return_rate=max_return_rate,
        )

    def extract_from_databases(self) -> Dict[str, Any]:
        return {
            "products": self._db.get_all_products(),
            "reviews": self._db.get_all_reviews(),
        }

    def filter_and_transform(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        all_products: List[Product] = raw_data["products"]
        valid_products: List[Product] = []
        filtered_out_ids: List[str] = []
        rejection_reasons: Dict[str, str] = {}

        for prod in all_products:
            is_valid, reason = self._filter_chain.evaluate(prod)
            if is_valid:
                valid_products.append(prod)
            else:
                filtered_out_ids.append(prod.product_id)
                rejection_reasons[prod.product_id] = reason or "Failed quality filter"

        return {
            "all_count": len(all_products),
            "valid_products": valid_products,
            "filtered_out_ids": filtered_out_ids,
            "rejection_reasons": rejection_reasons,
        }

    def ingest_to_s3(self, filtered_data: Dict[str, Any]) -> Dict[str, str]:
        valid_products: List[Product] = filtered_data["valid_products"]
        s3_records = [
            {
                "product_id": p.product_id,
                "title": p.title,
                "description": p.description,
                "content_type": p.content_type.value,
                "tags": p.tags,
                "avg_rating": p.avg_rating,
                "price": p.price,
            }
            for p in valid_products
        ]
        s3_data_uri = self._s3.put_object(
            bucket=self.DATA_BUCKET,
            key=self.FILTERED_PRODUCTS_KEY,
            data={
                "products": s3_records,
                "filtered_out_ids": filtered_data["filtered_out_ids"],
                "rejection_reasons": filtered_data["rejection_reasons"],
            },
        )
        return {"s3_data_uri": s3_data_uri}

    def cluster_and_publish_artifacts(
        self,
        filtered_data: Dict[str, Any],
        s3_staging_uris: Dict[str, str],
    ) -> Dict[str, Any]:
        valid_products: List[Product] = filtered_data["valid_products"]
        filtered_out_ids: List[str] = filtered_data["filtered_out_ids"]

        corpus = [p.to_text_corpus() for p in valid_products]
        vectorizer = TFIDFVectorizer(max_features=256)
        feature_matrix = vectorizer.fit_transform(corpus)

        k = min(self.num_clusters, max(1, len(valid_products)))
        kmeans = SphericalKMeans(n_clusters=k, max_iter=60, random_state=42)
        kmeans.fit(feature_matrix)

        product_to_cluster: Dict[str, int] = {}
        cluster_to_products: Dict[int, List[str]] = defaultdict(list)
        product_vectors: Dict[str, List[float]] = {}

        for idx, prod in enumerate(valid_products):
            cluster_id = int(kmeans.labels_[idx])
            product_to_cluster[prod.product_id] = cluster_id
            cluster_to_products[cluster_id].append(prod.product_id)
            product_vectors[prod.product_id] = np.round(feature_matrix[idx], 6).tolist()

        vocab, idf_weights = vectorizer.export_state()
        model_artifact = ProductClusterModel(
            model_version=f"v1.0-{int(datetime.now(timezone.utc).timestamp())}",
            num_clusters=k,
            vocabulary=vocab,
            idf_weights=idf_weights,
            centroids=np.round(kmeans.centroids_, 6).tolist(),
            product_to_cluster=product_to_cluster,
            cluster_to_products=dict(cluster_to_products),
            product_vectors=product_vectors,
            filtered_out_products=filtered_out_ids,
        )

        s3_model_uri = self._s3.put_object(
            bucket=self.MODEL_BUCKET,
            key=self.MODEL_KEY,
            data=model_artifact,
        )

        return {
            "status": "SUCCESS",
            "total_catalog_products": filtered_data["all_count"],
            "valid_products_ingested": len(valid_products),
            "bad_products_filtered_out": filtered_out_ids,
            "rejection_reasons": filtered_data["rejection_reasons"],
            "num_clusters_created": k,
            "cluster_distribution": {str(cid): len(pids) for cid, pids in cluster_to_products.items()},
            "s3_data_uri": s3_staging_uris["s3_data_uri"],
            "s3_model_uri": s3_model_uri,
            "model_version": model_artifact.model_version,
        }


