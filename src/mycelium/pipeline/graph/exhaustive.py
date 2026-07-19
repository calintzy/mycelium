"""
전수 조망 map-reduce 파워모드 (Phase 7.6) — DESIGN_GRAPHRAG §6, D-17.

`ask --exhaustive`에서만 호출되는 느린 경로다. 통합검색(상위 일부만 검색) 대신
**전 커뮤니티 요약을 빠짐없이** 훑어 전체 조망 답변을 만든다.

흐름:
  1. graph.gpickle 로드 → community_id별 멤버 버킷 재구성(summarize.py와 동일 로직).
  2. 멤버 구성 해시로 community_summaries.json 캐시에서 각 커뮤니티 요약을 매칭.
     (캐시 키가 멤버해시라 community_id↔요약을 그래프에서 재구성해 정확히 잇는다.)
  3. **map**: 각 커뮤니티 요약에 대해 질의 관련 부분답변 생성.
     무관하면 "관련 없음"으로 응답 → 조기 스킵(콜 수는 동일하나 reduce 컨텍스트 절감).
  4. **reduce**: 관련 부분답변들을 종합해 최종 조망 답변 + 근거 커뮤니티 목록.

비용: 커뮤니티 수만큼 map LLM콜 + reduce 1콜. 느려서 명시 플래그에서만 쓴다(D-17).
외부 네트워크 0(로컬 Ollama).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mycelium.adapters.graph_store import graph_exists, load_graph
from mycelium.adapters.llm import create_llm
from mycelium.core.config import Config
from mycelium.pipeline.graph.summarize import (
    _community_members,
    _load_summary_cache,
    _member_hash,
)

# map 단계에서 "이 커뮤니티는 질의와 무관"을 나타내는 정확 마커.
# LLM이 이 토큰을 출력하면 조기 스킵한다(reduce에서 제외).
_IRRELEVANT_MARKER = "관련 없음"

# map 단계 system 프롬프트 — 커뮤니티 요약 1개에 대한 부분답변 또는 무관 스킵.
_MAP_SYSTEM = (
    "너는 지식베이스의 주제 군집(커뮤니티) 요약 하나를 보고, 사용자 질의에 "
    "이 군집이 기여할 내용을 한국어로 2~4문장 부분답변으로 쓰는 비서다. "
    "이 군집이 질의와 무관하면 다른 말 없이 정확히 '관련 없음'이라고만 답하라. "
    "요약에 없는 내용을 지어내지 마라."
)

# reduce 단계 system 프롬프트 — 부분답변들을 종합해 전체 조망 답변.
_REDUCE_SYSTEM = (
    "너는 여러 주제 군집의 부분답변을 종합해 빠짐없는 전체 조망 답변을 쓰는 비서다. "
    "아래 [부분답변]들을 통합해 사용자 질의에 한국어로 답하라. "
    "중복은 합치고, 군집 간 공통 주제와 큰 그림을 드러내라. "
    "부분답변에 없는 내용을 새로 지어내지 마라."
)


@dataclass
class CommunityPartial:
    """map 단계 부분답변 1건 (커뮤니티 단위)."""

    community_id: int
    note_count: int
    answer: str  # 부분답변 텍스트(관련 없는 커뮤니티는 결과에 담지 않음)


@dataclass
class ExhaustiveResult:
    """전수 조망 map-reduce 결과 집계 (CLI 출력·검증용)."""

    final_answer: str  # reduce 종합 답변
    partials: list[CommunityPartial] = field(default_factory=list)  # 관련 부분답변들
    total_communities: int = 0  # 매칭된 전체 커뮤니티 수(map 대상)
    skipped_irrelevant: int = 0  # "관련 없음" 판정으로 조기 스킵된 수
    failed: int = 0  # LLM 호출 예외로 실패한 수(신뢰성 지표)
    no_graph: bool = False  # 그래프/요약 미존재 시 True

    @property
    def skipped(self) -> int:
        """하위 호환: skipped_irrelevant + failed 합계."""
        return self.skipped_irrelevant + self.failed


def _match_summaries(config: Config) -> dict[int, dict]:
    """
    graph.gpickle + 요약 캐시에서 community_id → {summary, note_count} 매핑을 만든다.

    캐시(community_summaries.json)는 멤버해시 키라 community_id를 직접 담지 않는다.
    그래서 그래프에서 community_id별 멤버를 재구성(summarize.py와 동일)하고,
    같은 멤버해시로 캐시에서 요약을 역조회해 정확히 잇는다.
    노트가 없는 순수 엔티티 군집은 요약 대상이 아니므로 제외(summarize.py와 동일 정책).
    """
    if not graph_exists(config):
        return {}

    graph = load_graph(config)
    buckets = _community_members(graph)
    cache = _load_summary_cache(config)

    matched: dict[int, dict] = {}
    for cid, members in sorted(buckets.items()):
        if members["note_count"] == 0:
            continue
        key = _member_hash(members["titles"], members["entities"])
        summary = cache.get(key)
        if not summary:
            # 캐시 미스(멤버 구성 변동 등) — 해당 커뮤니티는 조망에서 빠짐(재요약 필요).
            continue
        matched[cid] = {
            "summary": summary,
            "note_count": members["note_count"],
        }
    return matched


def exhaustive_overview(
    question: str,
    config: Config | None = None,
) -> ExhaustiveResult:
    """
    전 커뮤니티 요약을 map-reduce로 종합해 전체 조망 답변을 생성한다 (Phase 7.6).

    Parameters
    ----------
    question : str
        사용자 질의(전체 주제·조망형 질문에 적합).
    config : Config | None
        설정 객체. None이면 기본 Config().

    Returns
    -------
    ExhaustiveResult
        final_answer: reduce 종합 답변.
        partials: 관련 있다고 판정된 커뮤니티 부분답변 목록.
        total_communities / skipped: map 통계.
        no_graph: 그래프·요약이 없으면 True(친절 안내용).
    """
    cfg = config or Config()

    matched = _match_summaries(cfg)
    if not matched:
        return ExhaustiveResult(
            final_answer="",
            partials=[],
            total_communities=0,
            skipped_irrelevant=0,
            failed=0,
            no_graph=True,
        )

    llm = create_llm(cfg)

    # --- map: 각 커뮤니티 요약 → 부분답변(무관 스킵) ---
    partials: list[CommunityPartial] = []
    skipped_irrelevant = 0  # "관련 없음" 판정 수 — 내용 없음(정상 동작)
    failed = 0              # LLM 호출 예외 수 — 오류(신뢰성 지표)
    total = len(matched)

    for cid in sorted(matched.keys()):
        info = matched[cid]
        prompt = (
            f"[질의] {question}\n\n"
            f"[커뮤니티 {cid} 요약]\n{info['summary']}\n\n"
            "이 군집이 질의에 기여할 내용을 2~4문장으로 쓰거나, 무관하면 '관련 없음'만 답하라."
        )
        try:
            resp = llm.invoke([("system", _MAP_SYSTEM), ("human", prompt)])
            text = (
                resp.content if isinstance(resp.content, str) else str(resp.content)
            ).strip()
        except Exception as e:  # noqa: BLE001 — 커뮤니티 1개 실패가 전체를 막지 않게
            print(f"    [경고] 커뮤니티 {cid} map 실패, 스킵 — {e}")
            failed += 1
            continue

        # 무관 커뮤니티 조기 스킵 — 마커로 시작하거나 동일하면 reduce에서 제외.
        if text == _IRRELEVANT_MARKER or text.startswith(_IRRELEVANT_MARKER):
            skipped_irrelevant += 1
            continue

        partials.append(
            CommunityPartial(
                community_id=cid,
                note_count=int(info["note_count"]),
                answer=text,
            )
        )

    # 관련 부분답변이 하나도 없으면 reduce 없이 근거 없음 처리.
    if not partials:
        return ExhaustiveResult(
            final_answer="노트에 근거가 없습니다",
            partials=[],
            total_communities=total,
            skipped_irrelevant=skipped_irrelevant,
            failed=failed,
            no_graph=False,
        )

    # --- reduce: 부분답변 종합 ---
    parts_text = "\n\n".join(
        f"[커뮤니티 {p.community_id} (노트 {p.note_count}개)] {p.answer}"
        for p in partials
    )
    reduce_prompt = f"[질의] {question}\n\n[부분답변]\n{parts_text}\n\n위를 종합해 답하라."
    try:
        resp = llm.invoke([("system", _REDUCE_SYSTEM), ("human", reduce_prompt)])
        final = (
            resp.content if isinstance(resp.content, str) else str(resp.content)
        ).strip()
    except Exception as e:  # noqa: BLE001 — reduce 실패 시 부분답변 나열로 폴백
        print(f"    [경고] reduce 실패 — 부분답변 나열로 대체. {e}")
        final = parts_text

    return ExhaustiveResult(
        final_answer=final,
        partials=partials,
        total_communities=total,
        skipped_irrelevant=skipped_irrelevant,
        failed=failed,
        no_graph=False,
    )
