import uuid

from fastapi import APIRouter, Request
from fastapi.sse import EventSourceResponse

from app.features.agent.v1.graph import invoke_agent, stream_agent
from app.features.chats.v1.schemas.req import (
    ChatRequest,
)
from app.features.chats.v1.schemas.res import (
    InitSessionResponse,
)

chat_router = APIRouter()


@chat_router.post("/session/new", response_model=InitSessionResponse)
async def get_new_session_id():
    new_id = str(uuid.uuid4())
    return {"session_id": new_id}


@chat_router.post("/session/invoke")
async def chat_invoke(request_obj: Request, request: ChatRequest):
    response = await invoke_agent(
        request.user_question,
        request.session_id,
        workflow=request_obj.app.state.workflow,
    )
    return {"response": response}


@chat_router.post("/session/stream")
async def chat_stream(request_obj: Request, request: ChatRequest):
    return EventSourceResponse(
        stream_agent(
            request.user_question,
            request.session_id,
            workflow=request_obj.app.state.workflow,
        )
    )
