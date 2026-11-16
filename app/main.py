import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

from app.features.agent.v1.agent import create_workflow
from chainlit.utils import mount_chainlit
from fastapi import Depends, FastAPI
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.router import v1_router
from app.core.config import settings
from app.core.database.session import get_session
from app.core.exceptions import DatabaseConnectionError
from app.core.logging import setup_logging
from app.core.middleware import (
    register_cors_middleware,
    register_exception_handlers,
    register_logging_middleware,
    register_rate_limiter,
    register_trusted_hosts_middleware,
)
from app.core.schemas import ApiResponse

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = MemorySaver()
    app.state.workflow = create_workflow(checkpointer=checkpointer)
    yield


app = FastAPI(title=settings.api_name, version=settings.api_version)

app.include_router(v1_router, prefix="/api/v1")
mount_chainlit(
    app=app,
    target="frontend/frontend.py",
    path="/chat",
)

register_cors_middleware(app)
register_trusted_hosts_middleware(app)
register_rate_limiter(app)
register_exception_handlers(app)
register_logging_middleware(app)


@app.get("/")
def index():
    return ApiResponse(
        data={
            "api_info": {
                "name": settings.api_name,
                "version": settings.api_version,
            }
        }
    )


@app.get("/health")
async def health_check(session: Annotated[AsyncSession, Depends(get_session)]):
    """
    Verifies that the API is running and the database connection works
    """
    try:
        await asyncio.wait_for(session.execute(text("SELECT 1")), timeout=5.0)
        return ApiResponse(data={"database": "connected"})
    except (TimeoutError, SQLAlchemyError):
        raise DatabaseConnectionError
