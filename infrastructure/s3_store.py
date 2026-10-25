"""Simulated AWS S3 Object Store for Spark ETL staging datasets and MLlib K-Means model artifacts."""
import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from infrastructure.interfaces import IObjectStore


class InMemoryS3Store(IObjectStore):
    """Simulates S3 buckets and keys used by Spark jobs and the Recommendation Service."""

    def __init__(self) -> None:
        self._buckets: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def put_object(self, bucket: str, key: str, data: Any) -> str:
        if bucket not in self._buckets:
            self._buckets[bucket] = {}
        self._buckets[bucket][key] = {
            "payload": copy.deepcopy(data),
            "last_modified": datetime.now(timezone.utc).isoformat(),
        }
        return f"s3://{bucket}/{key}"

    def get_object(self, bucket: str, key: str) -> Optional[Any]:
        bucket_store = self._buckets.get(bucket, {})
        entry = bucket_store.get(key)
        if entry is None:
            return None
        return copy.deepcopy(entry["payload"])

    def list_keys(self, bucket: str, prefix: str = "") -> List[str]:
        bucket_store = self._buckets.get(bucket, {})
        return [k for k in bucket_store.keys() if k.startswith(prefix)]