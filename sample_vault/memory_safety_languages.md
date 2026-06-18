---
date: 2024-03-20
type: tech-note
public: true
---

# 메모리 안전성 언어 비교

메모리 안전(memory safety)은 dangling pointer, use-after-free, buffer overflow 같은
결함을 언어 차원에서 막는 성질이다.

## 접근 방식 분류
- **가비지 컬렉션**: Go, Java, Python — 런타임이 회수, 편하지만 GC pause.
- **소유권 모델**: Rust — 컴파일 타임에 검증, 런타임 비용 없음. [[rust_ownership_basics]] 참고.
- **수동 관리**: C, C++ — 자유롭지만 위험. 개발자가 모든 책임을 진다.

## 왜 중요한가
보안 취약점의 큰 비중이 메모리 안전 결함에서 나온다. Microsoft, Google 모두
신규 시스템 코드에 Rust 채택을 늘리는 이유다.

ownership 모델의 세부는 [[rust_ownership_basics]]에서 다룬다.
