# Paper Playground

> AI/ML 논문의 핵심 메커니즘을 인터랙티브 학습물로 바꾸는 한국어 Paper Playground 실험.

첫 실험: `Attention Is All You Need` - Self-Attention 편.

[English README](README.en.md)

## 왜 만들었나

논문 요약 도구는 이미 많다. 이 프로젝트는 요약보다 한 단계 더 좁고 깊은 목표를 잡는다.

```text
논문을 읽기 쉽게 요약하기 X
논문의 핵심 메커니즘을 만지고 실험할 수 있는 학습물로 바꾸기 O
```

첫 번째 실험은 Transformer 논문, 그중에서도 Self-Attention이다. 토큰, query/key/value,
score, scaling, softmax, multi-head attention을 작은 숫자와 직접 조작 가능한 UI로
익히는 것이 목표다.

## 지금 할 수 있는 것

- 문장 프리셋을 고르고 token을 추가, 삭제, 섞을 수 있다.
- query token과 key token을 직접 선택한다.
- 선택한 key의 attention bias를 올리거나 낮춰 softmax 결과가 어떻게 바뀌는지 본다.
- attention matrix, Q/K/V 벡터, raw score, scaled score, softmax weight를 확인한다.
- `sqrt(d_k)` scaling과 decoder future masking을 켜고 끈다.
- 작은 toy simulation과 논문 기준 숫자(base/big model)를 같이 본다.

## 실행

`index.html`을 브라우저에서 열면 된다.

빌드, 백엔드, 패키지 매니저, API 키가 필요 없다.

## 프로젝트 구조

```text
index.html        앱 화면과 논문 순서 섹션
styles.css        다크 모드 대시보드 스타일
app.js            deterministic attention toy simulation
SPEC.md           Paper Playground 실험 명세
README.en.md      English README
docs/PROJECT.md   제품 노트와 다음 단계
```

## 포지셔닝

한 줄로 말하면:

> AI/ML 논문의 핵심 메커니즘을 인터랙티브 학습물로 바꾸는 한국어 Paper Playground.

조금 더 좁히면:

> 첫 실험은 Transformer Self-Attention이고, 목표는 여러 AI 논문으로 확장할 수 있는 변환 흐름을 찾는 것이다.

## 다음 단계

1. Self-Attention 범위를 더 날카롭게 좁힌 `concepts.md` 작성
2. 토큰 3개, 벡터 2~4차원짜리 `attention-example.json` 추가
3. Q/K/V -> score -> scaling -> softmax -> weighted sum 단계별 모드 강화
4. 수식과 UI 결과가 맞는지 검증하는 작은 reference script 추가
