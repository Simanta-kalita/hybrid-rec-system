"""Simulated Apache Kafka Event Streaming Broker for proactive async recommendation triggers."""
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List
from domain.events import KafkaEvent
from infrastructure.interfaces import IEventBroker


class InMemoryKafkaBroker(IEventBroker):
    """
    Simulates Kafka topics, consumer group subscriptions, and event processing.
    Supports both synchronous immediate dispatch (for low-latency worker mode) and queued batch consumption.
    """

    def __init__(self, auto_dispatch: bool = True) -> None:
        self.auto_dispatch = auto_dispatch
        self._queues: Dict[str, Deque[KafkaEvent]] = defaultdict(deque)
        self._subscribers: Dict[str, List[Callable[[KafkaEvent], None]]] = defaultdict(list)
        self._event_history: List[KafkaEvent] = []

    def publish(self, event: KafkaEvent) -> None:
        self._queues[event.topic].append(event)
        self._event_history.append(event)
        if self.auto_dispatch:
            self.process_pending_events(topic=event.topic)

    def subscribe(self, topic: str, handler: Callable[[KafkaEvent], None]) -> None:
        self._subscribers[topic].append(handler)

    def process_pending_events(self, topic: str = "recommendation-events") -> int:
        queue = self._queues[topic]
        handlers = self._subscribers.get(topic, [])
        processed_count = 0
        while queue:
            event = queue.popleft()
            for handler in handlers:
                handler(event)
            event.processed = True
            processed_count += 1
        return processed_count

    def get_event_log(self, limit: int = 50) -> List[KafkaEvent]:
        return list(reversed(self._event_history[-limit:]))
