import asyncio

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings
from app.core.database.session import async_engine, sync_engine

embeddings = FastEmbedEmbedding(model_name=settings.embedding_model)

vector_store = PGVectorStore(
    engine=sync_engine,
    async_engine=async_engine,
    table_name=settings.vector_collection_name,
    embed_dim=384,
    hybrid_search=True,
)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embeddings)

hybrid_retriever = index.as_retriever(
    vector_store_query_mode="hybrid",
    similarity_top_k=15,
    hybrid_top_k=15,
)

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=5,
)

score_cutoff = SimilarityPostprocessor(similarity_cutoff=0.5)


async def retrieve_documents(query: str) -> list[NodeWithScore]:
    nodes = await hybrid_retriever.aretrieve(query)
    if not nodes:
        return []

    query_bundle = QueryBundle(query)
    reranked_nodes = await asyncio.to_thread(
        reranker.postprocess_nodes,
        nodes,
        query_bundle=query_bundle,
    )
    return score_cutoff.postprocess_nodes(reranked_nodes, query_bundle=query_bundle)
