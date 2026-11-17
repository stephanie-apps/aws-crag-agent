import structlog
from langchain_tavily import TavilySearch

from app.features.agent.v1.schemas import DocumentWithSource
from app.features.documents.v1.retriever.vector_db import retrieve_documents

tavily_web_search = TavilySearch(max_results=3, search_depth="basic")


logger = structlog.get_logger()


async def web_search_tool(user_question: str) -> tuple:
    try:
        web_results = await tavily_web_search.ainvoke({"query": user_question})

        formatted_context_blocks = []
        documents_with_sources = []
        for index, result in enumerate(web_results, start=1):
            documents_with_sources.append(
                DocumentWithSource(
                    index=index,
                    source_type="web",
                    file=result.get("url", "Web Source {index}"),
                    content="",
                )
            )
            formatted_context_blocks.append(
                f"--- Web Context Block [{index} (web)] ---\n"
                f"Source URL: {result.get('source"Web Source {index}')}\n"
                f"Extracted Content: {result.get('content', '')}\n"
            )

        combined_web_context = "\n".join(formatted_context_blocks)
        return (documents_with_sources, combined_web_context)

    except (ConnectionError, TimeoutError, TypeError, ValueError, RuntimeError):
        logger.exception("Tavily Search Failed")
        combined_web_context = (
            "No real-time web search results could be retrieved due to system limits."
        )

    return ([], combined_web_context)


async def retriever_tool(user_question: str) -> tuple:
    formatted_context_list = []
    documents_with_sources = []
    try:
        nodes = await retrieve_documents(user_question)

        for index, node in enumerate(nodes, start=1):
            file = node.node.metadata.get("source", f"PDF Source  {index}")
            content = node.node.get_content()

            documents_with_sources.append(
                DocumentWithSource(
                    index=index,
                    source_type="pdf",
                    file=file,
                    content=content,
                )
            )
            formatted_context_list.append(
                f"--- Document Source [{index} (pdf)]: {file} \n{content}"
            )

        documents_with_sources_formatted_str = "\n\n".join(formatted_context_list)

        return (documents_with_sources, documents_with_sources_formatted_str)

    except (ConnectionError, TimeoutError, TypeError, ValueError, RuntimeError):
        logger.exception("Document retrieval failed")
        return ([], "No relevant documents could be retrieved due to system limits.")
