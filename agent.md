# RAG QA 설계 명세 (로컬 업로드, 테이블 중심, PySide6 GUI)

- **목표·범위**
  - 기능: 단일 문서 Q&A, 다중 문서 비교(N-way, 예: 여러 제품명 입력 후 성능 비교), 출처 표시/하이라이트.
  - 대상: 제품/부품 데이터시트(표 다수), KR·EN 유지.
  - 출력: 단일=불릿, 비교=표, 출처 태그 `[문서명, 페이지]`.
  - SLO: 정확도 ≥95%, Hallucination <5%, 복합 표 해석 성공률 ≥90%.

- **데이터 소스 (로컬 업로드)**
  - 입력: 사용자 PC에서 PDF/CSV/JSON 업로드 → 세션별 임시 워크스페이스 생성·정리.
  - 메타 필수: `product_name`/`model_id`, `version`, `region`(있는 경우) 저장 → 비교 시 매칭 키로 사용.
  - 도구: LlamaIndex document loaders. 초기 CLI/스크립트, 후속 PySide6 GUI.

- **전처리·데이터 핸들링 (테이블 품질 최적화)**
  - 파이프라인: Marker 1차(레이아웃/표) → camelot/tabula 표 보정 → pymupdf/pdfplumber 텍스트 → OCR tesseract(+layoutparser) → 단위 정규화 pint.
  - 단위 규칙: 표준 단위(SI)로 변환한 값과 원문 단위를 모두 컨텍스트에 포함.
  - 테이블 직렬화: 병합셀 분해, 헤더 계층 `col_path`, 단위 열 추가; Markdown 표 + YAML 메타(`section`, `page`, `col_types`, `units`, `table_flag`). 40행↑는 상위 10행 + 핵심 통계(최솟값/최댓값/평균) 요약 + 전체 CSV 메타.
  - 챙크: 섹션/테이블 단위 450±100 tokens, 오버랩 10%, 표 주변 설명 포함.
  - 메타: `doc_id`(세션), `component_type`, `version`, `section_path`, `table_flag`, `page`, `bbox`, `upload_session`, `markdown_blob`.
  - 품질 관리: 파싱 실패/열 불일치/단위 오류 로그; 실패율 >2% 시 검수 큐; Markdown 렌더 미리보기.
  - 도구: Marker, camelot, tabula, pdfplumber, pymupdf, pytesseract, layoutparser, pandas/polars, Great Expectations.

- **임베딩**
  - 선택: OpenAI `text-embedding-3-large`(API 전용).
  - 전략: Markdown 스니펫 임베딩, `table_flag` 메타; 배치 64 fp16, L2 normalize, Redis/In-memory 캐시.
  - 세션 스토어: 업로드 세션별 임시 Qdrant 컬렉션 생성, 24h 정리.
  - 도구: OpenAI Embeddings API, LlamaIndex embedding wrappers.

- **인덱스·스토어**
  - 엔진: Qdrant(로컬) HNSW 컬렉션 per session.
  - 필터: `doc_id`, `component_type`, `version`, `table_flag`, `upload_session`.
  - 하이브리드: Qdrant sparse/BM25 또는 간단 TF-IDF로 part number·키워드 보완.
  - 도구: Qdrant 로컬 서비스.

- **검색·리랭크 (CPU 전제)**
  - 전처리: 모델명 정규화, 단위 변환, 동의어 매핑(예: “소비전력” → “power consumption”); 질의에서 `product_name`/`model_id` 엔티티 추출 후 필터.
  - 1차: 벡터 top-k=12; 단일=해당 `doc_id`, 비교=질의로 지정된 여러 `product_name`/`model_id` 필터로 허용 문서 제한.
  - 테이블 가중치: `table_flag`=true 스코어×1.2.
  - 리랭크: CPU용 `bge-reranker-base` 또는 경량 `bge-reranker-large-lite` top-8→4; 필요 시 LLM rerank 옵션.
  - 출처: `doc_id/section/page/bbox/markdown_snippet` 유지 → 하이라이트 매핑.
  - Fallback: 스코어 <0.25 → Qdrant sparse/BM25 → 없으면 “정보 없음”.
  - 도구: LlamaIndex Retrieval, Qdrant hybrid search, sentence-transformers reranker(CPU).

- **생성·프롬프트**
  - 모델: GPT-4o(API).
  - 컨텍스트: Markdown 스니펫 + YAML 메타 동반(표준 단위와 원문 단위 모두 포함).
  - 템플릿: 시스템 “데이터시트 근거만 사용, 출처 `[문서명, 페이지]`, 모르면 거부, 언어=사용자 요청.” 단일=불릿+값+단위+출처; 비교=표 `항목 | 모델A | 모델B | … | Notes` + 차이 요약(제품 수에 따라 열 동적 생성).
  - 응답: `[1]` 태그 → `doc_id/page/bbox` 매핑; 토큰 ≤2200; 금칙어(내부 가격) 필터.

- **PySide6 GUI**
  - 업로드 UI: 파일 드롭/선택 → 세션 인덱스 생성 진행률 표시.
  - 비교 모드: 다수 문서 선택 후 질의(제품명/모델ID 매핑 UI 포함).
  - PDF 뷰어: Qt PDF/GraphicsView, 태그 클릭 시 `bbox` 오버레이 하이라이트.
  - 단계별 로그: 파싱/임베딩/검색 진행 표시.

- **SLO 측정 계획**
  - 정확도·할루시네이션: 200~300문항 평가세트(단일/비교/조건부/오류유도 포함), LLM-as-judge + 휴먼 샘플 검증으로 채점, Hallucination 비율 집계.
  - 복합 표 해석: 표 50개 이상 샘플링, 헤더/셀 매칭 정확도와 단위 보존율(허용오차 범위 내 숫자 매칭) 측정 → 목표 90% 이상.
