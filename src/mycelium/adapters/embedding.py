"""
임베딩 어댑터 — OllamaEmbeddings 생성 팩토리.
로컬 ↔ 클라우드 교체 지점: 이 파일만 수정하면 된다.
현재는 로컬 Ollama(bge-m3)만 지원.
"""

from langchain_ollama import OllamaEmbeddings

from mycelium.core.config import Config


def create_embeddings(config: Config) -> OllamaEmbeddings:
    """
    OllamaEmbeddings 인스턴스를 생성해 반환한다.
    임베딩 차원은 bge-m3 기준 1024 (자동 추론, 하드코딩 불필요).
    """
    return OllamaEmbeddings(
        model=config.embedding_model,
        base_url=config.ollama_base_url,
    )
