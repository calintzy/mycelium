"""
Leiden 커뮤니티 탐지 (Phase 7.3) — DESIGN_GRAPHRAG §5-3, D-12.

networkx → igraph 변환 → leidenalg.find_partition(ModularityVertexPartition, seed 고정)
→ 각 노드에 community_id 부여. 단일 레벨로 시작(과설계 금지).

실패 시(igraph/leidenalg 미설치 등) networkx greedy_modularity_communities로 폴백.
"""

from __future__ import annotations

import warnings

import networkx as nx

# Leiden 재현성을 위한 고정 시드.
_LEIDEN_SEED = 42


def detect_communities(graph: nx.Graph) -> tuple[int, float]:
    """
    그래프에 커뮤니티를 탐지하고 각 노드에 community_id 속성을 부여한다(in-place).

    Returns:
        (커뮤니티 수, modularity). 빈 그래프면 (0, 0.0).
    """
    if graph.number_of_nodes() == 0:
        return 0, 0.0

    try:
        return _leiden(graph)
    except Exception as e:  # noqa: BLE001 — Leiden 실패 시 폴백
        warnings.warn(
            f"[mycelium] Leiden 실패 — networkx greedy 폴백. 원인: {e}",
            stacklevel=2,
        )
        return _greedy_fallback(graph)


def _leiden(graph: nx.Graph) -> tuple[int, float]:
    """networkx → igraph → leidenalg 커뮤니티 탐지."""
    import igraph as ig
    import leidenalg

    # networkx 노드(문자열 id)를 igraph 정수 인덱스로 매핑.
    nodes = list(graph.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in graph.edges()]
    # 엣지 가중치(weight, 없으면 1.0)를 Leiden에 전달.
    weights = [float(graph[u][v].get("weight", 1.0)) for u, v in graph.edges()]

    g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)

    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.ModularityVertexPartition,
        weights=weights if weights else None,
        seed=_LEIDEN_SEED,
    )

    # 각 노드에 community_id 부여.
    for comm_id, members in enumerate(partition):
        for vertex_idx in members:
            graph.nodes[nodes[vertex_idx]]["community_id"] = comm_id

    return len(partition), float(partition.modularity)


def _greedy_fallback(graph: nx.Graph) -> tuple[int, float]:
    """networkx greedy_modularity_communities 폴백."""
    from networkx.algorithms.community import (
        greedy_modularity_communities,
        modularity,
    )

    communities = list(greedy_modularity_communities(graph, weight="weight"))
    for comm_id, members in enumerate(communities):
        for node in members:
            graph.nodes[node]["community_id"] = comm_id

    mod = modularity(graph, communities) if communities else 0.0
    return len(communities), float(mod)
