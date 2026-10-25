"""
Chain of Responsibility Pattern for Spark ETL Product Quality Filtering.
Filters out bad products with bad reviews or excessive return rates during the Spark Read-Filter-Ingest stage.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from domain.entities import Product


class IProductQualityFilter(ABC):
    """Handler interface in the Product Quality Filter Chain."""

    def __init__(self, next_filter: Optional["IProductQualityFilter"] = None) -> None:
        self._next = next_filter

    def set_next(self, next_filter: "IProductQualityFilter") -> "IProductQualityFilter":
        self._next = next_filter
        return next_filter

    def evaluate(self, product: Product) -> Tuple[bool, Optional[str]]:
        is_valid, reason = self._check(product)
        if not is_valid:
            return False, reason
        if self._next is not None:
            return self._next.evaluate(product)
        return True, None

    @abstractmethod
    def _check(self, product: Product) -> Tuple[bool, Optional[str]]:
        pass


class MinimumRatingFilter(IProductQualityFilter):
    """Rejects products whose average review rating falls below `min_rating`."""

    def __init__(self, min_rating: float = 3.2, next_filter: Optional[IProductQualityFilter] = None) -> None:
        super().__init__(next_filter)
        self.min_rating = min_rating

    def _check(self, product: Product) -> Tuple[bool, Optional[str]]:
        if product.review_count > 0 and product.avg_rating < self.min_rating:
            return False, f"avg_rating ({product.avg_rating}) < min_rating ({self.min_rating})"
        return True, None


class MaximumReturnRateFilter(IProductQualityFilter):
    """Rejects products with high customer return/defect rates."""

    def __init__(self, max_return_rate: float = 0.30, next_filter: Optional[IProductQualityFilter] = None) -> None:
        super().__init__(next_filter)
        self.max_return_rate = max_return_rate

    def _check(self, product: Product) -> Tuple[bool, Optional[str]]:
        if product.return_rate > self.max_return_rate:
            return False, f"return_rate ({product.return_rate}) > max_return_rate ({self.max_return_rate})"
        return True, None


class MetadataCompletenessFilter(IProductQualityFilter):
    """Ensures product has non-empty title, description, and tags for n-dimensional vectorization."""

    def _check(self, product: Product) -> Tuple[bool, Optional[str]]:
        if not product.title.strip() or not product.description.strip() or not product.tags:
            return False, "Incomplete content metadata (title, description, or tags missing)"
        return True, None


def build_default_quality_filter_chain(
    min_rating: float = 3.2,
    max_return_rate: float = 0.30,
) -> IProductQualityFilter:
    """Factory helper assembling the default Spark ETL quality filter chain."""
    chain = MetadataCompletenessFilter()
    chain.set_next(MinimumRatingFilter(min_rating=min_rating)).set_next(
        MaximumReturnRateFilter(max_return_rate=max_return_rate)
    )
    return chain