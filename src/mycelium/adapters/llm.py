"""
LLM 어댑터 — ChatOllama 생성 팩토리.
로컬 ↔ 클라우드 교체 지점: 이 파일만 수정하면 된다 (D-5).
현재는 로컬 Ollama(qwen2.5:14b)만 지원.
"""

from langchain_ollama import ChatOllama

from mycelium.core.config import Config


def create_llm(config: Config) -> ChatOllama:
    """
    ChatOllama 인스턴스를 생성해 반환한다.
    temperature=0 — 답변 재현성을 높이고 할루시네이션 억제.
    모델명은 config.generation_model에서 읽는다 (기본: 'qwen2.5:14b').
    """
    return ChatOllama(
        model=config.generation_model,
        base_url=config.ollama_base_url,
        temperature=0,
    )
