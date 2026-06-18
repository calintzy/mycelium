"""
벡터스토어 어댑터 — Chroma 생성 팩토리.
거리 메트릭: cosine (정규화 임베딩에 L2보다 적합).
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from mycelium.core.config import Config


def create_vectorstore(config: Config, embeddings: OllamaEmbeddings) -> Chroma:
    """
    Chroma 인스턴스를 생성해 반환한다.
    - 거리 메트릭: cosine (collection_metadata로 지정)
    - 영속 경로: config.chroma_path
    """
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.chroma_path),
        collection_metadata={"hnsw:space": "cosine"},
    )
