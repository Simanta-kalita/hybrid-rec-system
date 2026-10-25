"""FastAPI Application Entrypoint for the Production Hybrid Recommendation System."""
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import AsyncIterator

# Ensure package root is importable regardless of how main.py is invoked
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI
from api.dependencies import get_container
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Bootstraps seed data, executes Spark Content + Collaborative pipelines, and warms up KV Store on startup."""
    get_container()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hybrid Recommendation System (Content + Collaborative Filtering)",
        description=(
            "Production Low-Level Design (LLD) implementation combining:\n"
            "1. **ML Approach 1: Content Filtering [Exploitation]** (Spark ETL filtering bad reviews -> S3 -> "
            "MLlib K-Means Clustering -> Cosine Similarity Product Service)\n"
            "2. **ML Approach 2: Collaborative Filtering [Exploration]** (Spark Multi-Aspect User Clustering -> "
            "Neo4j Property Graph DB -> Missing-Edge Discovery & Weighted Rating Prediction)\n"
            "3. **Async Recommendation Generator & Proactive Kafka Exhaustion Trigger** "
            "(Redis KV Store `{user_id -> [products]}` replenished asynchronously via Kafka events)"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
