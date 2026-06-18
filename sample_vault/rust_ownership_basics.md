---
date: 2024-03-12
type: tech-note
public: true
---

# Rust 소유권 기본기

Rust의 핵심은 소유권(ownership) 모델이다. 가비지 컬렉터 없이 메모리 안전을 보장한다.

## 소유권 3원칙
- 모든 값은 단 하나의 소유자(owner)를 가진다.
- 소유자가 스코프를 벗어나면 값은 자동으로 해제된다(drop).
- 한 번에 하나의 가변 참조(mutable borrow)만 허용된다.

## 빌림(borrowing)과 라이프타임
참조는 빌림이며, 컴파일러의 borrow checker가 dangling reference를 차단한다.
라이프타임 어노테이션은 참조가 얼마나 오래 유효한지를 컴파일러에 알려준다.

## 함께 보기
메모리 관리 철학은 [[memory_safety_languages]]에서 다른 언어와 비교한다.
실제 CLI 도구를 만들 때의 패턴은 [[cli_design_principles]]를 참고.
