import os
import uuid

import chainlit as cl
from chainlit.types import ThreadDict

from app.features.agent.v1.graph import stream_agent
from app.features.agent.v1.schemas import DocumentWithSource
from app.main import app


@cl.on_chat_start
async def on_chat_start():
    """Configure session ID and workflow instance"""
    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)
    await cl.Message(content="Hello! I am ready to chat.").send()


async def ask_agent(prompt: str, session_id: uuid.UUID):
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    res = cl.Message(content="")
    retrieved_sources: list[DocumentWithSource] = []

    async with cl.Step(name="Thinking...", show_input=False) as status_step:
        async for chunk in stream_agent(
            user_question=prompt,
            session_id=session_id,
            with_status=True,
            workflow=app.state.workflow,
            callbacks=[cb],
        ):
            if chunk["type"] == "status":
                status_step.name = f"{chunk['content']}..."
                await status_step.update()

            elif chunk["type"] == "token":
                await status_step.remove()
                await res.stream_token(chunk["content"])

            elif chunk["type"] == "sources":
                retrieved_sources.extend(chunk["content"])

    await res.send()

    if retrieved_sources:
        elements = []
        footer_links = []

        for document in retrieved_sources:
            if document.source_type == "pdf":
                filename = os.path.basename(document.file)
                label = f"[{document.index} ({document.source_type})] {filename}"

                elements.append(
                    cl.Text(name=label, content=document.content, display="side")
                )
                footer_links.append(label)

            elif document.source_type == "web":
                footer_links.append(f"[{document.index} ({document.source_type})]")

        res.elements = elements
        res.content += "\n\n**Sources:**\n" + "\n".join(
            f"- {link}" for link in footer_links
        )
        await res.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Respond to user question"""
    session_id: uuid.UUID = uuid.UUID(cl.user_session.get("session_id"))
    await ask_agent(message.content, session_id)


@cl.on_chat_resume
async def on_chat_resume(session: ThreadDict):
    pass

@cl.data_layer
def get_data_layer():
    return None