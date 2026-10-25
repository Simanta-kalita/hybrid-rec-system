"""Kafka domain events for proactive recommendation generation and user activity tracking."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid
from domain.enums import EventType


@dataclass
class KafkaEvent:
    """Envelope for messages published to the Kafka event bus."""
    event_type: EventType
    user_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:10]}")
    topic: str = "recommendation-events"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed: bool = False
    trigger_reason: Optional[str] = None
