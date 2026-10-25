"""Dependency Injection container & realistic production seed data for the Hybrid Recommendation System."""
from typing import Optional
from domain.entities import BrowsingEvent, Order, Product, Review, User
from domain.enums import ContentType
from infrastructure.graph_db import Neo4jMemoryGraphDB
from infrastructure.kafka_broker import InMemoryKafkaBroker
from infrastructure.kv_store import RedisRecommendationKVStore
from infrastructure.relational_db import InMemoryRelationalDB
from infrastructure.s3_store import InMemoryS3Store
from ml_pipelines.collaborative_pipeline import CollaborativeFilteringSparkPipeline
from ml_pipelines.content_pipeline import ContentFilteringSparkPipeline
from services.collaborative_service import CollaborativeFilteringService
from services.content_service import ContentFilteringService
from services.generator_worker import AsyncRecommendationGenerator
from services.recommendation_facade import HybridRecommendationFacade


class SystemContainer:
    """Singleton Dependency Injection container holding all repositories, pipelines, and services."""

    def __init__(self) -> None:
        self.relational_db = InMemoryRelationalDB()
        self.s3_store = InMemoryS3Store()
        self.graph_db = Neo4jMemoryGraphDB(random_seed=42)
        self.kv_store = RedisRecommendationKVStore()
        self.kafka_broker = InMemoryKafkaBroker(auto_dispatch=True)

        self.content_pipeline = ContentFilteringSparkPipeline(
            relational_db=self.relational_db,
            s3_store=self.s3_store,
            num_clusters=4,
            min_rating_threshold=3.2,
            max_return_rate=0.30,
        )
        self.collaborative_pipeline = CollaborativeFilteringSparkPipeline(
            relational_db=self.relational_db,
            s3_store=self.s3_store,
            graph_db=self.graph_db,
            min_rating_threshold=3.2,
            max_return_rate=0.30,
        )

        self.content_service = ContentFilteringService(
            relational_db=self.relational_db,
            s3_store=self.s3_store,
        )
        self.collaborative_service = CollaborativeFilteringService(
            graph_db=self.graph_db,
            relational_db=self.relational_db,
        )
        self.generator_worker = AsyncRecommendationGenerator(
            relational_db=self.relational_db,
            kv_store=self.kv_store,
            content_service=self.content_service,
            collaborative_service=self.collaborative_service,
            kafka_broker=self.kafka_broker,
            batch_size=6,
        )
        self.facade = HybridRecommendationFacade(
            relational_db=self.relational_db,
            graph_db=self.graph_db,
            kv_store=self.kv_store,
            kafka_broker=self.kafka_broker,
            content_service=self.content_service,
            collaborative_service=self.collaborative_service,
            generator_worker=self.generator_worker,
            exhaustion_watermark=2,
        )

    def seed_and_bootstrap(self) -> dict:
        """Populates relational tables, runs both Spark ML pipelines, and warms up the KV store."""
        products = [
            # --- Electronics Cluster ---
            Product(
                product_id="prod_iphone_16_pro",
                title="Apple iPhone 16 Pro Max",
                description="Flagship Apple iOS smartphone with A18 Pro chip, titanium body, and 5G camera system",
                content_type=ContentType.ELECTRONICS,
                tags=["apple", "smartphone", "ios", "flagship", "camera", "5g"],
                price=1199.0,
                avg_rating=4.9,
                review_count=420,
                return_rate=0.02,
            ),
            Product(
                product_id="prod_macbook_air_m3",
                title="Apple MacBook Air M3 Laptop",
                description="Lightweight Apple macOS laptop with M3 silicon chip, Retina display, and MagSafe",
                content_type=ContentType.ELECTRONICS,
                tags=["apple", "laptop", "macos", "flagship", "magsafe"],
                price=1299.0,
                avg_rating=4.9,
                review_count=310,
                return_rate=0.01,
            ),
            Product(
                product_id="prod_airpods_pro_2",
                title="Apple AirPods Pro 2 Wireless Earbuds",
                description="Active noise cancelling Apple iOS wireless bluetooth earbuds with MagSafe case",
                content_type=ContentType.ELECTRONICS,
                tags=["apple", "audio", "ios", "wireless", "magsafe", "bluetooth"],
                price=249.0,
                avg_rating=4.8,
                review_count=510,
                return_rate=0.03,
            ),
            Product(
                product_id="prod_magsafe_charger",
                title="Apple MagSafe Fast Wireless Charger",
                description="Magnetic fast wireless charger for Apple iPhone and AirPods iOS ecosystem",
                content_type=ContentType.ELECTRONICS,
                tags=["apple", "smartphone", "ios", "wireless", "magsafe", "charger"],
                price=39.0,
                avg_rating=4.6,
                review_count=190,
                return_rate=0.04,
            ),
            # --- Books Cluster ---
            Product(
                product_id="prod_ddia_book",
                title="Designing Data-Intensive Applications",
                description="Comprehensive engineering book on distributed systems, replication, partitioning, and batch/stream processing",
                content_type=ContentType.BOOKS,
                tags=["distributed-systems", "databases", "architecture", "engineering", "spark", "kafka"],
                price=45.0,
                avg_rating=4.95,
                review_count=850,
                return_rate=0.01,
            ),
            Product(
                product_id="prod_sys_design_vol1",
                title="System Design Interview Volume 1",
                description="Step-by-step guide to scalable distributed systems architecture and backend engineering design",
                content_type=ContentType.BOOKS,
                tags=["distributed-systems", "architecture", "scalability", "engineering", "backend"],
                price=38.0,
                avg_rating=4.8,
                review_count=620,
                return_rate=0.02,
            ),
            Product(
                product_id="prod_db_internals",
                title="Database Internals: Deep Dive into Distributed Data Systems",
                description="Storage engines, B-trees, LSM-trees, and distributed database replication architecture book",
                content_type=ContentType.BOOKS,
                tags=["distributed-systems", "databases", "architecture", "storage-engine", "engineering"],
                price=49.0,
                avg_rating=4.85,
                review_count=290,
                return_rate=0.01,
            ),
            Product(
                product_id="prod_ml_sys_design",
                title="Designing Machine Learning Systems",
                description="Iterative framework for production ML engineering, recommendation systems, feature stores, and MLOps",
                content_type=ContentType.BOOKS,
                tags=["machine-learning", "recommender", "architecture", "engineering", "mlops"],
                price=44.0,
                avg_rating=4.9,
                review_count=410,
                return_rate=0.01,
            ),
            # --- Videos & Songs Cluster ---
            Product(
                product_id="prod_recsys_video",
                title="System Design: Production Recommendation Engine Masterclass",
                description="Deep dive video lecture on Content Filtering, Collaborative Filtering, Spark K-Means, Neo4j, and Kafka",
                content_type=ContentType.VIDEOS,
                tags=["machine-learning", "recommender", "distributed-systems", "kafka", "spark", "neo4j"],
                price=29.0,
                avg_rating=4.9,
                review_count=180,
                return_rate=0.0,
            ),
            Product(
                product_id="prod_lld_video",
                title="Low-Level Design & Clean Architecture in Python Masterclass",
                description="Hands-on engineering video course covering SOLID principles, design patterns, and FastAPI microservices",
                content_type=ContentType.VIDEOS,
                tags=["architecture", "engineering", "backend", "lld", "design-patterns"],
                price=29.0,
                avg_rating=4.85,
                review_count=150,
                return_rate=0.0,
            ),
            Product(
                product_id="prod_lofi_beats",
                title="Midnight Coding Synthwave & Lo-Fi Beats",
                description="Instrumental ambient electronic and synthwave music album for deep programming focus",
                content_type=ContentType.SONGS,
                tags=["music", "synthwave", "ambient", "focus", "instrumental"],
                price=9.99,
                avg_rating=4.8,
                review_count=220,
                return_rate=0.0,
            ),
            Product(
                product_id="prod_cyber_ambient",
                title="Deep Space Ambient Electronica Mix",
                description="Atmospheric instrumental synthwave and ambient focus tracks for late-night reading and coding",
                content_type=ContentType.SONGS,
                tags=["music", "synthwave", "ambient", "focus", "instrumental"],
                price=9.99,
                avg_rating=4.75,
                review_count=140,
                return_rate=0.0,
            ),
            # --- Bad Products (Should be filtered out by the Spark ETL Job!) ---
            Product(
                product_id="prod_bad_knockoff_phone",
                title="Defective Clone Smartphone X99",
                description="Cheap smartphone with broken battery and fake camera sensors",
                content_type=ContentType.ELECTRONICS,
                tags=["smartphone", "camera", "5g"],
                price=89.0,
                avg_rating=1.9,
                review_count=75,
                return_rate=0.52,
            ),
            Product(
                product_id="prod_bad_plagiarized_book",
                title="Unedited AI Dump Coding Pamphlet",
                description="Poorly formatted pamphlet with broken code snippets and missing pages",
                content_type=ContentType.BOOKS,
                tags=["engineering", "backend", "architecture"],
                price=19.0,
                avg_rating=2.1,
                review_count=40,
                return_rate=0.45,
            ),
        ]

        for p in products:
            self.relational_db.upsert_product(p)

        users = [
            User(
                user_id="user_target",
                name="Arjun (Target Multi-Aspect User)",
                preferred_categories=[ContentType.BOOKS, ContentType.ELECTRONICS],
            ),
            User(
                user_id="user_alice",
                name="Alice (Distributed Systems & ML Bookworm)",
                preferred_categories=[ContentType.BOOKS, ContentType.VIDEOS],
            ),
            User(
                user_id="user_bob",
                name="Bob (Apple Ecosystem Enthusiast)",
                preferred_categories=[ContentType.ELECTRONICS],
            ),
            User(
                user_id="user_chloe",
                name="Chloe (Full-Stack Architect & Synthwave Listener)",
                preferred_categories=[ContentType.BOOKS, ContentType.SONGS, ContentType.VIDEOS],
            ),
        ]
        for u in users:
            self.relational_db.upsert_user(u)

        # Orders demonstrating Multi-Aspect User Similarity:
        # - user_target bought DDIA (Books) and MacBook Air M3 (Electronics).
        # - user_alice bought DDIA, System Design Vol 1, Database Internals, and Designing ML Systems (similar in BOOKS!).
        # - user_bob bought MacBook Air M3, iPhone 16 Pro, AirPods Pro 2, MagSafe Charger (similar in ELECTRONICS!).
        # - user_chloe bought DDIA, System Design Vol 1, RecSys Video, and Lo-Fi Beats.
        orders = [
            Order("ord_1", "user_target", "prod_ddia_book", implicit_rating=5.0),
            Order("ord_2", "user_target", "prod_macbook_air_m3", implicit_rating=5.0),
            Order("ord_3", "user_alice", "prod_ddia_book", implicit_rating=5.0),
            Order("ord_4", "user_alice", "prod_sys_design_vol1", implicit_rating=4.9),
            Order("ord_5", "user_alice", "prod_db_internals", implicit_rating=4.8),
            Order("ord_6", "user_alice", "prod_ml_sys_design", implicit_rating=5.0),
            Order("ord_7", "user_bob", "prod_macbook_air_m3", implicit_rating=5.0),
            Order("ord_8", "user_bob", "prod_iphone_16_pro", implicit_rating=5.0),
            Order("ord_9", "user_bob", "prod_airpods_pro_2", implicit_rating=4.8),
            Order("ord_10", "user_bob", "prod_magsafe_charger", implicit_rating=4.7),
            Order("ord_11", "user_chloe", "prod_ddia_book", implicit_rating=4.9),
            Order("ord_12", "user_chloe", "prod_sys_design_vol1", implicit_rating=4.8),
            Order("ord_13", "user_chloe", "prod_recsys_video", implicit_rating=5.0),
            Order("ord_14", "user_chloe", "prod_lofi_beats", implicit_rating=4.9),
        ]
        for o in orders:
            self.relational_db.add_order(o)

        browsing_events = [
            BrowsingEvent("brw_1", "user_target", "prod_sys_design_vol1", dwell_time_seconds=140.0, interaction_type="read"),
            BrowsingEvent("brw_2", "user_target", "prod_iphone_16_pro", dwell_time_seconds=95.0, interaction_type="view"),
            BrowsingEvent("brw_3", "user_target", "prod_lofi_beats", dwell_time_seconds=180.0, interaction_type="listen"),
            BrowsingEvent("brw_4", "user_alice", "prod_recsys_video", dwell_time_seconds=210.0, interaction_type="watch"),
            BrowsingEvent("brw_5", "user_bob", "prod_iphone_16_pro", dwell_time_seconds=160.0, interaction_type="view"),
        ]
        for b in browsing_events:
            self.relational_db.add_browsing_event(b)

        reviews = [
            Review("rev_bad_1", "user_bob", "prod_bad_knockoff_phone", rating=1.5, comment="Overheats and screen flickers."),
            Review("rev_bad_2", "user_alice", "prod_bad_plagiarized_book", rating=2.0, comment="Terrible quality, avoid."),
        ]
        for r in reviews:
            self.relational_db.add_review(r)

        # Execute Spark Pipeline 1 (Content Filtering) & Spark Pipeline 2 (Collaborative Filtering)
        content_stats = self.content_pipeline.run_spark_job()
        collab_stats = self.collaborative_pipeline.run_spark_job()

        # Run initial Async Recommendation Generator job to populate Redis KV Store for all users
        kv_counts = self.generator_worker.generate_for_all_users()

        return {
            "content_pipeline": content_stats,
            "collaborative_pipeline": collab_stats,
            "kv_store_counts": kv_counts,
        }


_container_instance: Optional[SystemContainer] = None


def get_container() -> SystemContainer:
    """FastAPI dependency returning the initialized singleton SystemContainer."""
    global _container_instance
    if _container_instance is None:
        _container_instance = SystemContainer()
        _container_instance.seed_and_bootstrap()
    return _container_instance


def reset_container() -> SystemContainer:
    """Resets and re-seeds the singleton container (used in deterministic tests)."""
    global _container_instance
    _container_instance = SystemContainer()
    _container_instance.seed_and_bootstrap()
    return _container_instance