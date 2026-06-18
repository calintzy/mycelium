"""
인덱싱 파이프라인 — 볼트 마크다운을 청킹·임베딩해 Chroma에 적재한다.

흐름:
  볼트 .md 수집
  → frontmatter 파싱 (pyyaml)
  → MarkdownHeaderTextSplitter (1차, 헤더 단위)
  → RecursiveCharacterTextSplitter (2차, 임계 초과 섹션만)
  → 메타데이터 부착
  → Chroma 적재 (전체 재생성, D-6 v1)
"""

from __future__ import annotations

import re
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from mycelium.adapters.embedding import create_embeddings
from mycelium.adapters.vectorstore import create_vectorstore
from mycelium.core.artifacts import is_generated_artifact
from mycelium.core.config import Config
from mycelium.core.models import ChunkMetadata

# 인덱싱에서 제외할 디렉토리명 (숨김 디렉토리 + 특수 폴더)
_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".obsidian",
        ".omc",
        ".git",
        ".venv",
        "__pycache__",
        "00-INBOX",  # 미분류 임시 파일, 인덱싱 제외
    }
)

# MarkdownHeaderTextSplitter에 전달할 헤더 레벨 설정
_HEADER_LEVELS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


@dataclass
class IngestionResult:
    """인덱싱 결과 집계."""
    file_count: int
    chunk_count: int
    chroma_path: Path


def _default_chroma_path() -> Path:
    """
    config.py가 결정하는 기본 chroma 경로를 재현한다 (public_only 발 밟기 가드용).
    CHROMA_PATH 환경변수가 있으면 그것이 기본, 없으면 <프로젝트>/chroma.
    config.Config의 default_factory 로직과 동일해야 한다 (단일 진실 유지).
    """
    import os

    env = os.environ.get("CHROMA_PATH")
    if env:
        return Path(env)
    # config.py 기준: Path(<core/config.py>).parents[3] / "chroma" == <프로젝트>/chroma.
    # 여기(pipeline/ingestion.py)도 parents[3]가 동일 프로젝트 루트다.
    return Path(__file__).parents[3] / "chroma"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    YAML frontmatter를 파싱해 (메타데이터 딕셔너리, 본문) 튜플로 반환한다.
    frontmatter가 없으면 ({}, 원문) 반환.
    """
    # frontmatter는 파일 첫 줄이 '---'로 시작하는 경우만 처리
    if not text.startswith("---"):
        return {}, text

    # 두 번째 '---' 위치 탐색
    end_match = re.search(r"\n---\s*\n", text[3:])
    if end_match is None:
        return {}, text

    fm_text = text[3 : end_match.start() + 3]
    body = text[end_match.start() + 3 + len(end_match.group()) :]

    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}

    return fm, body


def _is_public(frontmatter: dict) -> bool:
    """
    frontmatter의 public 필드가 명시적으로 True인지 판정한다 (D-7 deny-by-default).
    bool True 또는 "true"/"True" 문자열만 공개로 간주, 미표시·그 외는 비공개.
    """
    val = frontmatter.get("public", False)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def _build_header_path(metadata: dict) -> str:
    """
    MarkdownHeaderTextSplitter가 생성한 메타데이터에서
    헤더 계층 경로 문자열을 만든다.
    예: {"h1": "제목", "h2": "섹션"} → "제목 > 섹션"
    """
    parts = []
    for key in ("h1", "h2", "h3"):
        val = metadata.get(key)
        if val:
            parts.append(val.strip())
    return " > ".join(parts)


def _collect_md_files(
    vault_path: Path, config: Config | None = None
) -> Generator[Path, None, None]:
    """
    볼트 루트에서 .md 파일을 재귀 수집한다.
    _EXCLUDE_DIRS에 포함된 디렉토리는 건너뛴다.
    config가 주어지면 생성산출물(graph_report.md 등, D-11)도 함께 제외한다 —
    인덱싱·그래프 빌드가 동일 기준(core.artifacts)으로 제외하도록 단일 진실 공유.
    """
    cfg = config or Config()
    for item in vault_path.rglob("*.md"):
        # 경로 구성 요소 중 제외 디렉토리가 있으면 스킵
        if any(part in _EXCLUDE_DIRS for part in item.parts):
            continue
        # 생성산출물(graph_report.md·graph.json·.graphify*)은 메인 인덱스·그래프에서 제외.
        if is_generated_artifact(item, cfg):
            continue
        yield item


def _split_document(
    file_path: Path,
    vault_path: Path,
    frontmatter: dict,
    body: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """
    마크다운 본문을 청킹해 Document 리스트로 반환한다.
    1차: MarkdownHeaderTextSplitter (헤더 단위)
    2차: RecursiveCharacterTextSplitter (임계 초과 섹션만)
    """
    # 볼트 루트 기준 상대 경로
    source = str(file_path.relative_to(vault_path))

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADER_LEVELS,
        strip_headers=False,  # 헤더 텍스트를 본문에 남겨 검색 품질 유지
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 1차 분할
    header_chunks: list[Document] = header_splitter.split_text(body)

    result: list[Document] = []
    # 파일 단위로 증가하는 청크 인덱스 — chunk_id 고유성 보장 (C2).
    # 2차분할 sub-chunk마다, 헤더없는 노트의 청크마다 다른 값을 부여한다.
    chunk_index = 0
    for doc in header_chunks:
        header_path = _build_header_path(doc.metadata)

        # 임계 초과 시 2차 분할 (각 sub-chunk가 별개의 chunk_index를 갖게 함)
        if len(doc.page_content) > chunk_size:
            pieces = char_splitter.split_text(doc.page_content)
        else:
            pieces = [doc.page_content]

        for piece in pieces:
            chunk_id = f"{source}::{header_path}::{chunk_index}"
            meta = ChunkMetadata.from_frontmatter(
                source=source,
                header_path=header_path,
                frontmatter=frontmatter,
                chunk_id=chunk_id,
            ).to_chroma_dict()
            result.append(Document(page_content=piece, metadata=meta))
            chunk_index += 1

    # 본문이 완전히 비어있는 파일은 검색 코퍼스에서 제외(스킵) — 더미 '(empty)' 청크 미생성.
    return result


def run_ingestion(
    config: Config | None = None, public_only: bool = False
) -> IngestionResult:
    """
    볼트 전체를 인덱싱해 Chroma에 적재한다.
    재실행 시 기존 컬렉션을 삭제 후 재생성 (전체 재인덱싱, D-6 v1).

    Parameters
    ----------
    config : Config | None
        설정 객체. None이면 기본 Config().
    public_only : bool
        True면 frontmatter `public: true`인 노트의 청크만 적재한다 (D-7 deny-by-default).
        공개 코퍼스는 실볼트 인덱스와 섞이지 않도록 별도 chroma 경로(--chroma)로 분리해야 한다.

    Returns:
        IngestionResult — 파일 수, 청크 수, Chroma 경로
    """
    if config is None:
        config = Config()

    # 공개 인덱싱 발 밟기 가드 — public_only=True인데 chroma_path가 기본 경로
    # (<프로젝트>/chroma, 실볼트 전체 인덱스)면 실인덱스를 공개 코퍼스로 덮어쓸 위험.
    # 기본 경로면 명확한 에러로 별도 경로 지정을 요구한다 (실인덱스 보호, D-7).
    if public_only and config.chroma_path.resolve() == _default_chroma_path().resolve():
        raise ValueError(
            "공개 전용 인덱싱(public_only)이 기본 chroma 경로를 덮어쓰려 합니다. "
            "실볼트 인덱스 보호를 위해 --chroma로 별도 경로를 지정하세요 "
            f"(현재 경로: {config.chroma_path})."
        )

    # 임베딩·벡터스토어 생성
    embeddings: OllamaEmbeddings = create_embeddings(config)

    # 기존 컬렉션 삭제 후 재생성 (전체 재인덱싱)
    _reset_collection(config, embeddings)
    vectorstore: Chroma = create_vectorstore(config, embeddings)

    vault_path = config.vault_path
    if not vault_path.exists():
        raise FileNotFoundError(f"볼트 경로가 존재하지 않습니다: {vault_path}")

    all_docs: list[Document] = []
    file_count = 0

    for md_file in _collect_md_files(vault_path, config):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"  [경고] 파일 읽기 실패, 건너뜀: {md_file} — {e}")
            continue

        frontmatter, body = _parse_frontmatter(text)

        # 공개 전용 모드(D-7 deny-by-default): public:true 노트가 아니면 건너뜀.
        if public_only and not _is_public(frontmatter):
            continue

        chunks = _split_document(
            file_path=md_file,
            vault_path=vault_path,
            frontmatter=frontmatter,
            body=body,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        all_docs.extend(chunks)
        file_count += 1

    # Chroma 문서 ID로 안정적 chunk_id 사용 (C2 — dense/BM25 공통 키와 일치).
    # 드물게 chunk_id가 겹치면(동일 source/header/index) UUID로 충돌 회피.
    ids: list[str] = []
    seen_ids: set[str] = set()
    for doc in all_docs:
        cid = doc.metadata.get("chunk_id") or str(uuid.uuid4())
        if cid in seen_ids:
            cid = f"{cid}::{uuid.uuid4()}"
            doc.metadata["chunk_id"] = cid
        seen_ids.add(cid)
        ids.append(cid)

    # 배치 적재 (문서가 많을 경우 Chroma 내부에서 자동 배치 처리)
    if all_docs:
        vectorstore.add_documents(documents=all_docs, ids=ids)

    return IngestionResult(
        file_count=file_count,
        chunk_count=len(all_docs),
        chroma_path=config.chroma_path,
    )


def _reset_collection(config: Config, embeddings: OllamaEmbeddings) -> None:
    """
    기존 Chroma 컬렉션을 삭제한다.
    chroma_path가 없거나 컬렉션이 없으면 조용히 넘어간다.
    """
    import chromadb

    chroma_path = config.chroma_path
    if not chroma_path.exists():
        return

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        client.delete_collection(config.collection_name)
    except Exception as e:
        # "컬렉션 없음"만 정상으로 간주해 무시, 그 외 오류는 경고로 노출.
        msg = str(e).lower()
        if "does not exist" in msg or "not found" in msg:
            return
        warnings.warn(
            f"[mycelium] 컬렉션 삭제 중 예기치 못한 오류 (계속 진행): {e}",
            stacklevel=2,
        )
