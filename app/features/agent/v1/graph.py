import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.features.agent.v1.nodes import nodes
from app.features.agent.v1.routers import (
    intent_router,
    is_input_safe_router,
    is_output_safe_router,
)
from app.features.agent.v1.state import GraphState


def create_workflow(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    for node in nodes:
        graph.add_node(node)

    graph.add_edge(START, "input_guardrail_node")
    graph.add_conditional_edges("input_guardrail_node", is_input_safe_router)
    graph.add_conditional_edges("intent_classifier_node", intent_router)
    graph.add_edge("query_rewriter_node", "retriever_node")
    graph.add_edge("retriever_node", "generator_node")
    graph.add_edge("generator_node", "output_guardrail_node")
    graph.add_conditional_edges("output_guardrail_node", is_output_safe_router)
    graph.add_edge("fallback_node", END)

    return graph.compile(checkpointer=checkpointer)



async def invoke_agent(
    user_question: str,
    session_id: uuid.UUID 
    callbacks: list[Any] | None = None,
    workflow: CompiledStateGraph, 
    return_state: bool = False,
) -> str | dict[str, Any]:
    config: RunnableConfig = {
        "configurable": {"thread_id": str(session_id)},
    }
    if callbacks:
        config["callbacks"] = callbacks

    inputs: Any = {
        "user_question": user_question,
        "messages": [HumanMessage(content=user_question)],
    }

    response = await workflow.ainvoke(input=inputs, config=config)
    if return_state:
        return response
    return response["messages"][-1].content


async def stream_agent(
    user_question: str,
    session_id: uuid.UUID,
    callbacks: list[Any] | None = None,
    with_status: bool = False,
    workflow: CompiledStateGraph,
) -> AsyncGenerator[dict[str, Any], None]:
    config: RunnableConfig = {
        "configurable": {"thread_id": str(session_id)},
    }
    if callbacks:
        config["callbacks"] = callbacks

    inputs: Any = {
        "user_question": user_question,
        "messages": [HumanMessage(content=user_question)],
    }

    stream_modes: list[
        Literal[
            "values",
            "updates",
            "checkpoints",
            "tasks",
            "debug",
            "messages",
            "custom",
        ]
    ] = ["messages", "updates"]
    if with_status:
        stream_modes.append("custom")

    RESPONSE_NODE_NAMES = ("generator_node", "fallback_node")

    async for chunk_type, chunk_data in workflow.astream(
        input=inputs,
        config=config,
        stream_mode=stream_modes,
    ):
        if chunk_type == "messages" and isinstance(chunk_data, tuple):
            msg, metadata = chunk_data

            if (
                metadata.get("langgraph_node") in RESPONSE_NODE_NAMES
                and isinstance(msg, AIMessageChunk)
                and msg.content
            ):
                text = ""
                if isinstance(msg.content, str):
                    text = msg.content
                elif isinstance(msg.content, list):
                    text = "".join(
                        block.get("text", "")
                        for block in msg.content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                    )

                if text:
                    yield {"type": "token", "content": text}

        elif chunk_type == "updates" and isinstance(chunk_data, dict):
            for node_name, node_output in chunk_data.items():
                if (
                    isinstance(node_output, dict)
                    and "documents_with_sources" in node_output
                ):
                    yield {
                        "type": "sources",
                        "content": node_output["documents_with_sources"],
                    }

            if "fallback_node" in chunk_data:
                node_output = chunk_data["fallback_node"]
                fallback_text = node_output.get("synthesis_response")
                if fallback_text:
                    yield {"type": "token", "content": fallback_text}

        elif chunk_type == "custom":
            yield {"type": "status", "content": str(chunk_data)}
