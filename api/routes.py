"""FastAPI REST Controllers exposing Content Filtering, Collaborative Filtering, Hybrid KV Feed, and Kafka Triggers."""
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Query
from api.dependencies import SystemContainer, get_container
from api.schemas import (
    ConsumeFeedRequest,
    ConsumeFeedResponse,
    KafkaTriggerRequest,
    PipelineExecutionResponse,
    ProductCreateRequest,
    ProductInteractionRequest,
    ProductInteractionResponse,
    ProductResponse,
    RecommendationItemSchema,
    UserFeedResponse,
)
from domain.entities import Product
from domain.enums import CandidateSelectionStrategy
from domain.events import KafkaEvent

router = APIRouter(prefix="/api/v1")


@router.get("/health", tags=["System"])
def health_check(container: SystemContainer = Depends(get_container)) -> Dict[str, Any]:
    """Returns system health, S3 model keys, Graph DB summary, and Kafka event log stats."""
    return {
        "status": "healthy",
        "s3_data_lake_keys": container.s3_store.list_keys("rec-data-lake"),
        "s3_model_keys": container.s3_store.list_keys("rec-models"),
        "graph_db_summary": container.graph_db.get_graph_summary(),
        "recent_kafka_events": len(container.kafka_broker.get_event_log()),
    }


@router.get("/products", response_model=List[ProductResponse], tags=["Catalog"])
def list_products(container: SystemContainer = Depends(get_container)) -> List[Product]:
    """Lists all catalog products (including ratings and return rates)."""
    return container.relational_db.get_all_products()


@router.post("/products", response_model=ProductResponse, status_code=201, tags=["Catalog"])
def create_product(
    req: ProductCreateRequest,
    container: SystemContainer = Depends(get_container),
) -> Product:
    """Adds a new product to the Relational Product DB."""
    product = Product(
        product_id=req.product_id,
        title=req.title,
        description=req.description,
        content_type=req.content_type,
        tags=req.tags,
        price=req.price,
        avg_rating=req.avg_rating,
        review_count=req.review_count,
        return_rate=req.return_rate,
    )
    container.relational_db.upsert_product(product)
    return product


@router.post("/pipelines/run", response_model=PipelineExecutionResponse, tags=["Spark ML Pipelines"])
def trigger_spark_pipelines(
    container: SystemContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    Executes both Spark + MLlib batch pipelines:
      1. Content Filtering Spark Job (Read/Filter bad products -> S3 -> K-Means Product Clusters -> S3 Model)
      2. Collaborative Filtering Spark Job (User/Order/Browsing ETL -> S3 -> Multi-Aspect User K-Means -> Neo4j Graph DB)
      3. Async Recommendation Generator batch run to refresh the KV Store for all users.
    """
    content_res = container.content_pipeline.run_spark_job()
    collab_res = container.collaborative_pipeline.run_spark_job()
    kv_counts = container.generator_worker.generate_for_all_users()
    return {
        "content_pipeline_result": content_res,
        "collaborative_pipeline_result": collab_res,
        "kv_store_precomputed_counts": kv_counts,
    }


@router.get(
    "/recommendations/content/{product_id}",
    response_model=List[RecommendationItemSchema],
    tags=["ML Approach 1: Content Filtering (Exploitation)"],
)
def get_content_filtering_recommendations(
    product_id: str,
    top_k: int = Query(default=5, ge=1, le=20),
    container: SystemContainer = Depends(get_container),
) -> Any:
    """
    When a user is on a product page or transacts, matches the product to its K-Means cluster
    and returns the top-K most similar products ranked by Cosine Similarity (excluding bad products).
    """
    product = container.relational_db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")
    return container.content_service.recommend_similar_to_product(product_id=product_id, top_k=top_k)


@router.get(
    "/recommendations/collaborative/{user_id}",
    response_model=List[RecommendationItemSchema],
    tags=["ML Approach 2: Collaborative Filtering (Exploration)"],
)
def get_collaborative_filtering_recommendations(
    user_id: str,
    strategy: CandidateSelectionStrategy = Query(default=CandidateSelectionStrategy.TOP_N_SIMILAR),
    top_n_users: int = Query(default=5, ge=1, le=20),
    top_k_items: int = Query(default=5, ge=1, le=20),
    container: SystemContainer = Depends(get_container),
) -> Any:
    """
    Executes Graph DB Collaborative Filtering for a user:
      1. Queries candidate users from the user's multi-aspect clusters using `strategy`
         (`random_sample`, `top_n_similar`, or `all_if_small`).
      2. Traverses `[:PURCHASED]` edges to find candidate items bought by candidate users but NOT by `user_id`.
      3. Predicts user interest rating weighted by user-user similarity and returns top-K exploratory items.
    """
    user = container.relational_db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return container.collaborative_service.recommend_exploratory_items(
        user_id=user_id,
        strategy=strategy,
        top_n_users=top_n_users,
        top_k_items=top_k_items,
    )


@router.get(
    "/recommendations/feed/{user_id}",
    response_model=UserFeedResponse,
    tags=["Hybrid KV Store Serving & Kafka Trigger"],
)
def get_user_recommendation_feed(
    user_id: str,
    container: SystemContainer = Depends(get_container),
) -> Any:
    """
    Fetches precomputed hybrid recommendations (`Exploitation + Exploration`) from the Recommendation KV Store
    (`{ user_id -> [product_1, product_2, ...] }`). If empty, triggers Kafka event to regenerate.
    """
    user = container.relational_db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return container.facade.get_user_feed(user_id=user_id)


@router.post(
    "/recommendations/feed/{user_id}/consume",
    response_model=ConsumeFeedResponse,
    tags=["Hybrid KV Store Serving & Kafka Trigger"],
)
def consume_user_recommendations(
    user_id: str,
    req: ConsumeFeedRequest,
    container: SystemContainer = Depends(get_container),
) -> Any:
    """
    Marks `consume_count` items from the user's KV recommendation list as consumed.
    When remaining unconsumed recommendations drop to or below the exhaustion watermark,
    automatically pushes a `RECOMMENDATIONS_EXHAUSTED` event to Kafka so the Async Recommendation Generator
    proactively replenishes the KV Store!
    """
    user = container.relational_db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    return container.facade.consume_user_feed(
        user_id=user_id,
        consume_count=req.consume_count,
        strategy=req.strategy,
    )


@router.post(
    "/interactions",
    response_model=ProductInteractionResponse,
    tags=["Real-Time Interactions & Proactive Events"],
)
def record_interaction(
    req: ProductInteractionRequest,
    container: SystemContainer = Depends(get_container),
) -> Any:
    """
    Records a user transaction (`purchase`) or content consumption (`view`, `watch`, `read`, `listen`),
    returns immediate Content Filtering recommendations for the product page, and publishes a Kafka event
    to refresh the user's KV Store feed asynchronously.
    """
    if not container.relational_db.get_user(req.user_id):
        raise HTTPException(status_code=404, detail=f"User '{req.user_id}' not found")
    if not container.relational_db.get_product(req.product_id):
        raise HTTPException(status_code=404, detail=f"Product '{req.product_id}' not found")
    return container.facade.record_product_interaction(
        user_id=req.user_id,
        product_id=req.product_id,
        interaction_type=req.interaction_type,
        rating=req.rating,
        dwell_time_seconds=req.dwell_time_seconds,
    )


@router.post("/events/kafka-trigger", tags=["Hybrid KV Store Serving & Kafka Trigger"])
def emit_kafka_regeneration_event(
    req: KafkaTriggerRequest,
    container: SystemContainer = Depends(get_container),
) -> Dict[str, Any]:
    """Explicitly publishes an event from the frontend to Kafka to force async recommendation regeneration."""
    event = KafkaEvent(
        event_type=req.event_type,
        user_id=req.user_id,
        payload={
            "strategy": req.strategy.value,
            "exploitation_ratio": req.exploitation_ratio,
        },
        trigger_reason="Explicit frontend Kafka trigger",
    )
    container.kafka_broker.publish(event)
    updated_feed = container.kv_store.get_recommendations(req.user_id, include_consumed=False)
    return {
        "event_id": event.event_id,
        "topic": event.topic,
        "processed": event.processed,
        "regenerated_feed_size": len(updated_feed),
    }


@router.get("/graph/users/{user_id}/cohorts", tags=["Graph DB Inspection"])
def inspect_user_graph_cohorts(
    user_id: str,
    strategy: CandidateSelectionStrategy = Query(default=CandidateSelectionStrategy.TOP_N_SIMILAR),
    container: SystemContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    Inspects the user's multi-aspect cluster memberships in the Graph DB (e.g., books vs. electronics cohorts),
    candidate similar users, and missing-edge candidate products.
    """
    user = container.relational_db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")

    memberships = container.graph_db.get_user_clusters(user_id)
    candidate_users = container.graph_db.query_candidate_users(user_id=user_id, strategy=strategy, top_n=5)
    candidate_items = container.graph_db.query_candidate_items(
        target_user_id=user_id,
        candidate_users=candidate_users,
    )
    return {
        "user_id": user_id,
        "multi_aspect_cluster_memberships": [
            {
                "cluster_id": m.cluster_id,
                "aspect": m.aspect,
                "similarity_score": m.similarity_score,
            }
            for m in memberships
        ],
        "candidate_similar_users": [
            {"candidate_user_id": uid, "similarity": sim, "shared_aspect": aspect}
            for uid, sim, aspect in candidate_users
        ],
        "missing_edge_candidate_products": {
            pid: [
                {"endorsed_by": cuid, "user_similarity": sim, "rating": rating, "aspect": aspect}
                for cuid, sim, rating, aspect in endorsements
            ]
            for pid, endorsements in candidate_items.items()
        },
    }


@router.get("/events/log", tags=["System"])
def get_kafka_event_log(container: SystemContainer = Depends(get_container)) -> List[Dict[str, Any]]:
    """Returns recent Kafka events processed by the Async Recommendation Generator."""
    events = container.kafka_broker.get_event_log(limit=30)
    return [
        {
            "event_id": e.event_id,
            "topic": e.topic,
            "event_type": e.event_type.value,
            "user_id": e.user_id,
            "trigger_reason": e.trigger_reason,
            "processed": e.processed,
            "timestamp": e.timestamp.isoformat(),
            "payload": e.payload,
        }
        for e in events
    ]
