# 과학 지식 검색 RAG 시스템

과학 분야 문서를 색인하고 질의에 대해 가장 관련성 높은 문단을 검색하는 RAG 경진대회 프로젝트입니다.

**결과: MAP 0.9447 / MRR 0.9455, Mid LB 1위 / 14팀**

## 핵심 접근

### 검색 파이프라인
```
질의 → 동의어 확장 (science_synonyms.txt)
     → BGE-M3 Dense 임베딩 (Qdrant)
     → BM25 Sparse 검색
     → RRF 앙상블 (Dense + Sparse)
     → Qwen3-8B Reranker (LoRA 파인튜닝)
     → 최종 Top-K 결과
```

### 자체 검증 설계 (GT 없는 환경)
Ground Truth 없이 7일간 16가지 전략을 비교하기 위해 자체 검증 기준을 설계했습니다:
- **SHA256 해시 비교**: 같은 전략이 같은 결과를 내는지 재현성 확인
- **Top-3 변화량 추적**: 전략 변경 시 상위 결과가 얼마나 달라지는지 모니터링

### Reranker 파인튜닝
- 모델: Qwen3-8B (LoRA)
- SFT 데이터: 과학 도메인 triplet 3,000+ 쌍 (`data/reranker_triplets_sample.jsonl`)
- 실패 기록: negatives 오탐 226개 → 학습 데이터 필터링 후 개선

## 파일 구성

```
rag-science-search/
├── src/
│   └── rag_pipeline.py     # 메인 RAG 파이프라인 (Dense+Sparse+Reranker)
├── notebooks/
│   └── experiments.ipynb   # 전략별 실험 기록
└── data/
    ├── science_synonyms.txt # 과학 분야 동의어 사전
    └── user_dict.txt        # 커스텀 토크나이저 사전
```

## 기술 스택

| 구분 | 내용 |
|------|------|
| Dense 임베딩 | BAAI/bge-m3 |
| 벡터 DB | Qdrant |
| Sparse 검색 | BM25 |
| Reranker | Qwen3-8B (LoRA fine-tuned) |
| 파인튜닝 | Unsloth + TRL |
| 평가 | MAP@10, MRR@10 |
