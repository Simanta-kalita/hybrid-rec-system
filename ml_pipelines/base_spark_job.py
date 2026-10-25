"""
Template Method Pattern for Spark Batch ETL & MLlib Clustering Jobs.
Defines the invariant 4-step pipeline skeleton shared by Content Filtering and Collaborative Filtering jobs:
  1. extract_from_databases()
  2. filter_and_transform()
  3. ingest_to_s3()
  4. cluster_and_publish_artifacts()
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class AbstractSparkClusteringJob(ABC):
    """Template Method base class for Spark Read-Filter-Ingest + MLlib K-Means pipelines."""

    def run_spark_job(self) -> Dict[str, Any]:
        """Template method executing the deterministic 4-stage Spark pipeline."""
        raw_data = self.extract_from_databases()
        filtered_data = self.filter_and_transform(raw_data)
        s3_staging_uris = self.ingest_to_s3(filtered_data)
        clustering_summary = self.cluster_and_publish_artifacts(filtered_data, s3_staging_uris)
        return clustering_summary

    @abstractmethod
    def extract_from_databases(self) -> Dict[str, Any]:
        """Stage 1: Read raw records from Relational DBs (Product DB, Users DB, Orders DB, Browsing History)."""
        pass

    @abstractmethod
    def filter_and_transform(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Filter out bad records (e.g., low-rated/high-return products) and construct feature payloads."""
        pass

    @abstractmethod
    def ingest_to_s3(self, filtered_data: Dict[str, Any]) -> Dict[str, str]:
        """Stage 3: Persist cleaned datasets into S3 data lake buckets."""
        pass

    @abstractmethod
    def cluster_and_publish_artifacts(
        self,
        filtered_data: Dict[str, Any],
        s3_staging_uris: Dict[str, str],
    ) -> Dict[str, Any]:
        """Stage 4: Execute MLlib K-Means clustering and publish models to S3 / Graph DB."""
        pass
