"""In-memory Relational Database adapter simulating Products DB, Users DB, Orders DB, and Browsing History DB."""
from typing import Dict, List, Optional
from domain.entities import BrowsingEvent, Order, Product, Review, User
from infrastructure.interfaces import IRelationalRepository


class InMemoryRelationalDB(IRelationalRepository):
    """Thread-safe in-memory store for transactional tables (Products, Users, Orders, Reviews, Browsing)."""

    def __init__(self) -> None:
        self._products: Dict[str, Product] = {}
        self._users: Dict[str, User] = {}
        self._orders: List[Order] = []
        self._reviews: List[Review] = []
        self._browsing_events: List[BrowsingEvent] = []

    def get_all_products(self) -> List[Product]:
        return list(self._products.values())

    def get_product(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)

    def upsert_product(self, product: Product) -> None:
        self._products[product.product_id] = product

    def get_all_users(self) -> List[User]:
        return list(self._users.values())

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def upsert_user(self, user: User) -> None:
        self._users[user.user_id] = user

    def get_all_orders(self) -> List[Order]:
        return list(self._orders)

    def get_user_orders(self, user_id: str) -> List[Order]:
        return [o for o in self._orders if o.user_id == user_id]

    def add_order(self, order: Order) -> None:
        self._orders.append(order)

    def get_all_reviews(self) -> List[Review]:
        return list(self._reviews)

    def add_review(self, review: Review) -> None:
        self._reviews.append(review)
        # Recompute product average rating and review count dynamically
        product = self._products.get(review.product_id)
        if product:
            prod_reviews = [r.rating for r in self._reviews if r.product_id == review.product_id]
            product.review_count = len(prod_reviews)
            product.avg_rating = round(sum(prod_reviews) / len(prod_reviews), 2)

    def get_all_browsing_events(self) -> List[BrowsingEvent]:
        return list(self._browsing_events)

    def get_user_browsing_events(self, user_id: str) -> List[BrowsingEvent]:
        return [e for e in self._browsing_events if e.user_id == user_id]

    def add_browsing_event(self, event: BrowsingEvent) -> None:
        self._browsing_events.append(event)
