"""
생성 산출물 판정 — 단일 진실 (D-11 노이즈 제외).

graph_report.md·graph.json·.graphify* 같은 Graphify/자동생성 파일은 실측상 노이즈
허브(deg58)다. 인덱싱(ingestion)·그래프 빌드(graph/build·enrich) 양쪽이 동일 기준으로
제외해야 하므로 판정 로직을 core에 두어 단일 진실로 공유한다 (의존방향: pipeline→core).
"""

from __future__ import annotations

from pathlib import Path

from mycelium.core.config import Config


def is_generated_artifact(path: Path, config: Config) -> bool:
    """
    생성 산출물(Graphify/자동생성)인지 판정한다 (D-11 노이즈 제외).
    - 파일명(소문자)이 graph_exclude_files에 있으면 제외.
    - 경로 구성요소 중 graph_exclude_prefixes(.graphify 등)로 시작하는 것이 있으면 제외.
    """
    if path.name.lower() in config.graph_exclude_files:
        return True
    for part in path.parts:
        for prefix in config.graph_exclude_prefixes:
            if part.startswith(prefix):
                return True
    return False
