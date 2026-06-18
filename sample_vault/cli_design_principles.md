---
date: 2024-04-02
type: tech-note
public: true
---

# CLI 도구 설계 원칙

좋은 커맨드라인 인터페이스(CLI)는 발견 가능성과 일관성이 핵심이다.

## 핵심 가이드라인
- 인자가 없으면 도움말을 보여준다 (no_args_is_help).
- 사람이 읽을 출력과 기계가 파싱할 출력(`--json`)을 분리한다.
- 종료 코드(exit code)를 의미 있게 쓴다: 0은 성공, 비0은 실패.
- 긴 작업에는 진행 로그를 출력한다.

## 서브커맨드 패턴
`tool verb` 형태(예: `git commit`, `docker run`)가 확장에 유리하다.
Python에서는 typer나 click이 이 패턴을 잘 지원한다.

## 함께 보기
시스템 언어로 빠른 CLI를 만들 때는 [[rust_ownership_basics]]가 도움이 된다.
검색 기능을 붙인다면 [[hybrid_search_notes]]의 융합 기법을 참고.
