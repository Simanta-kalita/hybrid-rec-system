"""
Comprehensive Unit & End-to-End Integration Test Suite for the Hybrid Recommendation System.
"""
import asyncio
from pathlib import Path
import sys
from typing import Any
import unittest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import httpx
from fastapi.testclient import TestClient
from api.dependencies import reset_container
from domain.enums import CandidateSelectionStrategy, RecommendationSource
from main import create_app


class _ASGITestClient:
    """Fallback synchronous test client using httpx.ASGITransport (for httpx >= 0.28 with older Starlette)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async def _send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(_send())

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)


def build_test_client(app: Any) -> Any:
    """Creates a Starlette TestClient, or falls back to _ASGITestClient if httpx >= 0.28 rejects `app=`."""
    try:
        return TestClient(app)
    except TypeError:
        return _ASGITestClient(app)


class TestHybridRecommendationSystemE2E(unittest.TestCase):
    def setUp(self) -> None:
        self.container = reset_container()
        self.app = create_app()
        self.client = build_test_client(self.app)

    def test_01_spark_content_pipeline_filters_bad_products_and_creates_s3_model(self) -> None:
        """Verifies Spark ETL filters out bad products with poor reviews before K-Means clustering."""
        model = self.container.s3_store.get_object("rec-models", "content/kmeans_product_model.json")
        self.assertIsNotNone(model)
        self.assertIn("prod_bad_knockoff_phone", model.filtered_out_products)
        self.assertIn("prod_bad_plagiarized_book", model.filtered_out_products)
        self.assertNotIn("prod_bad_knockoff_phone", model.product_to_cluster)
        self.assertNotIn("prod_bad_plagiarized_book", model.product_to_cluster)
        self.assertIn("prod_iphone_16_pro", model.product_to_cluster)
        self.assertIn("prod_ddia_book", model.product_to_cluster)

    def test_02_content_filtering_exploitation_on_product_page(self) -> None:
        """Verifies Content Filtering recommends items from the same K-Means cluster using Cosine Similarity."""
        resp = self.client.get("/api/v1/recommendations/content/prod_iphone_16_pro?top_k=3")
        self.assertEqual(resp.status_code, 200)
        items = resp.json()
        self.assertEqual(len(items), 3)

        rec_ids = [item["product_id"] for item in items]
        self.assertNotIn("prod_bad_knockoff_phone", rec_ids)
        for item in items:
            self.assertEqual(item["source"], RecommendationSource.CONTENT_FILTERING.value)
            self.assertEqual(item["content_type"], "electronics")
            self.assertGreater(item["score"], 0.3)

    def test_03_multi_aspect_user_clustering_and_graph_db_traversal(self) -> None:
        """Verifies that `user_target` belongs to multiple aspect clusters in the Graph DB."""
        cohort_resp = self.client.get("/api/v1/graph/users/user_target/cohorts")
        self.assertEqual(cohort_resp.status_code, 200)
        cohort_data = cohort_resp.json()

        aspects = {m["aspect"] for m in cohort_data["multi_aspect_cluster_memberships"]}
        self.assertIn("books", aspects)
        self.assertIn("electronics", aspects)
        self.assertGreaterEqual(len(cohort_data["multi_aspect_cluster_memberships"]), 2)

        for strategy in [
            CandidateSelectionStrategy.TOP_N_SIMILAR,
            CandidateSelectionStrategy.RANDOM_SAMPLE,
            CandidateSelectionStrategy.ALL_IF_SMALL,
        ]:
            cf_resp = self.client.get(
                f"/api/v1/recommendations/collaborative/user_target?strategy={strategy.value}&top_k_items=5"
            )
            self.assertEqual(cf_resp.status_code, 200)
            cf_items = cf_resp.json()
            self.assertGreater(len(cf_items), 0)
            cf_pids = {item["product_id"] for item in cf_items}
            self.assertNotIn("prod_ddia_book", cf_pids)
            self.assertNotIn("prod_macbook_air_m3", cf_pids)
            self.assertTrue(
                bool(cf_pids.intersection({"prod_iphone_16_pro", "prod_sys_design_vol1", "prod_db_internals", "prod_airpods_pro_2"}))
            )

    def test_04_kv_store_serving_and_proactive_kafka_exhaustion_trigger(self) -> None:
        """Verifies that consuming recommendations from the KV Store triggers a proactive Kafka event."""
        feed_resp = self.client.get("/api/v1/recommendations/feed/user_target")
        self.assertEqual(feed_resp.status_code, 200)
        initial_feed = feed_resp.json()
        self.assertEqual(initial_feed["unconsumed_count"], 6)
        self.assertEqual(initial_feed["regeneration_cycles"], 1)

        c1_resp = self.client.post(
            "/api/v1/recommendations/feed/user_target/consume",
            json={"consume_count": 2, "strategy": "top_n_similar"},
        )
        self.assertEqual(c1_resp.status_code, 200)
        c1_data = c1_resp.json()
        self.assertEqual(len(c1_data["consumed_items"]), 2)
        self.assertEqual(c1_data["remaining_before_kafka_trigger"], 4)
        self.assertFalse(c1_data["proactive_kafka_triggered"])
        self.assertEqual(c1_data["remaining_after_regeneration"], 4)

        c2_resp = self.client.post(
            "/api/v1/recommendations/feed/user_target/consume",
            json={"consume_count": 3, "strategy": "top_n_similar"},
        )
        self.assertEqual(c2_resp.status_code, 200)
        c2_data = c2_resp.json()
        self.assertEqual(len(c2_data["consumed_items"]), 3)
        self.assertEqual(c2_data["remaining_before_kafka_trigger"], 1)
        self.assertTrue(c2_data["proactive_kafka_triggered"])
        self.assertIsNotNone(c2_data["kafka_event_id"])
        self.assertEqual(c2_data["remaining_after_regeneration"], 6)
        self.assertEqual(c2_data["total_regenerations_for_user"], 2)

    def test_05_real_time_product_interaction_updates_graph_and_triggers_kafka(self) -> None:
        """Verifies purchasing/viewing a product returns immediate content recs and emits a Kafka refresh event."""
        inter_resp = self.client.post(
            "/api/v1/interactions",
            json={
                "user_id": "user_target",
                "product_id": "prod_lofi_beats",
                "interaction_type": "purchase",
                "rating": 4.9,
            },
        )
        self.assertEqual(inter_resp.status_code, 200)
        data = inter_resp.json()
        self.assertEqual(data["product_id"], "prod_lofi_beats")
        self.assertTrue(data["kafka_event_id"].startswith("evt_"))
        self.assertGreater(len(data["immediate_content_recommendations"]), 0)
        sim_pids = [item["product_id"] for item in data["immediate_content_recommendations"]]
        self.assertIn("prod_cyber_ambient", sim_pids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
