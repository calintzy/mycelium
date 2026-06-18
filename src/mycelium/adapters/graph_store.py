"""
그래프 저장소 어댑터 — networkx 그래프 영속(gpickle 저장/로드).

DESIGN_GRAPHRAG §4: 지속화는 `graph/graph.gpickle`.
원문 파생물이라 .gitignore 대상(프라이버시).

저장 포맷으로 gpickle을 쓰는 이유:
  - networkx 네이티브, 노드/엣지 속성(dict)을 손실 없이 보존.
  - graphml은 중첩 dict·리스트 속성을 직렬화하지 못해 부적합(엔티티 관계 라벨·tags 보존 필요).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from mycelium.core.config import Config

# 그래프 파일명 (graph_path 디렉토리 하위에 저장).
_GRAPH_FILENAME = "graph.gpickle"


def _graph_file(config: Config) -> Path:
    """graph.gpickle 파일 경로를 반환한다."""
    return config.graph_path / _GRAPH_FILENAME


def save_graph(graph: nx.Graph, config: Config | None = None) -> Path:
    """
    networkx 그래프를 gpickle로 저장한다.
    graph_path 디렉토리가 없으면 생성한다.

    Returns:
        저장된 파일 경로.
    """
    cfg = config or Config()
    cfg.graph_path.mkdir(parents=True, exist_ok=True)
    path = _graph_file(cfg)
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_graph(config: Config | None = None) -> nx.Graph:
    """
    저장된 그래프를 로드해 반환한다.
    파일이 없으면 FileNotFoundError를 던진다(graph-build 미실행 안내용).
    """
    cfg = config or Config()
    path = _graph_file(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"그래프 파일이 없습니다: {path}. 먼저 `graph-build`를 실행하세요."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_exists(config: Config | None = None) -> bool:
    """그래프 파일이 존재하면 True (graph-build 실행 여부 확인용)."""
    cfg = config or Config()
    return _graph_file(cfg).exists()
