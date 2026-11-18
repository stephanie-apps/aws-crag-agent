import asyncio

from llama_index.core import VectorStoreIndex
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

from fastembed.rerank.cross_encoder import TextCrossEncoder
from llama_index.core.postprocessor import SimilarityPostprocessor

cross_encoder = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
score_cutoff = SimilarityPostprocessor(similarity_cutoff=0.5)


def _rerank_sync(
    query: str, nodes: list[NodeWithScore], top_n: int = 5
) -> list[NodeWithScore]:
    texts = [n.node.get_content() for n in nodes]
    scores = list(cross_encoder.rerank(query, texts))

    # calibrated reranker scores to each node and sort descending
    for node, score in zip(nodes, scores):
        node.score = float(score)

        ranked = sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
        return ranked[:top_n]


async def retrieve_documents(query: str) -> list[NodeWithScore]:
    nodes = await hybrid_retriever.aretrieve(query)
    if not nodes:
        return []

    reranked_nodes = await asyncio.to_thread(_rerank_sync, query, nodes, 5)
    return score_cutoff.postprocess_nodes(
        reranked_nodes, query_bundle=QueryBundle(query)
    )
