"""Pydantic v2 Request and Response DTO schemas for the FastAPI Hybrid Recommendation System."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from domain.enums import (
    CandidateSelectionStrategy,
    ContentType,
    EventType,
    RecommendationSource,
)


class ProductCreateRequest(BaseModel):
    product_id: str = Field(..., examples=["prod_headphone_xm5"])
    title: str = Field(..., examples=["Sony WH-1000XM5 Wireless Noise Canceling Headphones"])
    description: str = Field(..., examples=["Industry-leading ANC bluetooth over-ear headphones with 30hr battery"])
    content_type: ContentType = Field(default=ContentType.ELECTRONICS)
    tags: List[str] = Field(default_factory=list, examples=[["audio", "wireless", "anc", "bluetooth"]])
    price: float = Field(..., gt=0)
    avg_rating: float = Field(default=4.5, ge=0.0, le=5.0)
    review_count: int = Field(default=10, ge=0)
    return_rate: float = Field(default=0.02, ge=0.0, le=1.0)


class ProductResponse(BaseModel):
    product_id: str
    title: str
    description: str
    content_type: ContentType
    tags: List[str]
    price: float
    avg_rating: float
    review_count: int
    return_rate: float


class RecommendationItemSchema(BaseModel):
    product_id: str
    title: str
    content_type: ContentType
    score: float
    source: RecommendationSource
    explanation: str
    cluster_id: Optional[str] = None
    consumed: bool = False
    generated_at: datetime


class UserFeedResponse(BaseModel):
    user_id: str
    unconsumed_count: int
    proactive_kafka_triggered: bool
    regeneration_cycles: int
    recommendations: List[RecommendationItemSchema]


class ConsumeFeedRequest(BaseModel):
    consume_count: int = Field(default=3, ge=1, le=20, description="Number of recommendations user scrolled/consumed")
    strategy: CandidateSelectionStrategy = Field(default=CandidateSelectionStrategy.TOP_N_SIMILAR)


class ConsumeFeedResponse(BaseModel):
    user_id: str
    consumed_items: List[RecommendationItemSchema]
    remaining_before_kafka_trigger: int
    proactive_kafka_triggered: bool
    kafka_event_id: Optional[str] = None
    remaining_after_regeneration: int
    total_regenerations_for_user: int


class ProductInteractionRequest(BaseModel):
    user_id: str = Field(..., examples=["user_target"])
    product_id: str = Field(..., examples=["prod_iphone_16_pro"])
    interaction_type: str = Field(default="purchase", pattern="^(purchase|view|watch|read|listen)$")
    rating: float = Field(default=5.0, ge=1.0, le=5.0)
    dwell_time_seconds: float = Field(default=120.0, ge=1.0)


class ProductInteractionResponse(BaseModel):
    user_id: str
    product_id: str
    interaction_type: str
    kafka_event_id: str
    immediate_content_recommendations: List[RecommendationItemSchema]


class KafkaTriggerRequest(BaseModel):
    user_id: str = Field(..., examples=["user_target"])
    event_type: EventType = Field(default=EventType.RECOMMENDATIONS_EXHAUSTED)
    strategy: CandidateSelectionStrategy = Field(default=CandidateSelectionStrategy.TOP_N_SIMILAR)
    exploitation_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class PipelineExecutionResponse(BaseModel):
    content_pipeline_result: Dict[str, Any]
    collaborative_pipeline_result: Dict[str, Any]
    kv_store_precomputed_counts: Dict[str, int]
