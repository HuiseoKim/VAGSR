# VAGSR_Inference: Vector Augmented Generation with Semantic Retrieval

이 프로젝트는 벡터 증강 생성(Vector Augmented Generation)과 의미적 검색(Semantic Retrieval)을 결합한 RAG(Retrieval-Augmented Generation) 시스템의 훈련 및 추론 구현체입니다.

## 📁 프로젝트 구조

```
VAGSR_Inference/
├── train_code/                 # 모델 훈련 코드
├── inference_naive_RAG/        # 텍스트 기반 Naive RAG 추론
├── inference_code_noRAG/       # RAG 없는 순수 추론 
├── inference_code/             # 벡터 기반 RAG 추론 (메인)
└── data/                       # 데이터 저장소
    ├── task/                   # 태스크별 데이터셋
    ├── faiss/                  # FAISS 벡터 인덱스
    └── parameter/              # 모델 가중치 파일
```

## 🚀 주요 기능

### 1. 훈련 시스템 (`train_code/`)
- **기본 모델**: DeepSeek-R1-Distill-Llama-8B
- **훈련 방식**: 프로젝터(Projector) 레이어만 학습, 베이스 LLM은 frozen
- **특별 토큰**: `[CTX]` 토큰을 추가하여 벡터 임베딩을 LLM에 주입
- **메모리 효율성**: 그라디언트 체크포인팅, bfloat16 정밀도 사용

#### 주요 파일
- `train.py`: 메인 훈련 스크립트
- `model.py`: VectorRagModel 클래스 정의
- `trainer.py`: PyTorch Lightning 기반 훈련 로직
- `data.py`: 데이터 로딩 및 전처리
- `paraphrase_training.sh`: 훈련 실행 스크립트

### 2. 추론 시스템

#### A. `inference_code/` - 벡터 기반 RAG (메인 추론 시스템)
- **기능**: 확신도 기반 적응적 벡터 검색
- **검색 방식**: FAISS 인덱스를 통한 고속 벡터 유사도 검색
- **확신도 임계값**: 모델 확신도가 낮을 때만 RAG 수행하여 효율성 증대

#### B. `inference_naive_RAG/` - 텍스트 기반 Naive RAG
- **기능**: 전통적인 텍스트 기반 RAG 추론
- **검색 방식**: 텍스트 검색 후 컨텍스트로 추가

#### C. `inference_code_noRAG/` - RAG 없는 순수 추론
- **기능**: RAG 없이 순수 LLM 추론만 수행
- **용도**: 베이스라인 성능 비교

### 3. 데이터 (`data/`)
- **task/**: ASQA, BBEH 영화 추천, 스포츠 QA, Linguini 등 다양한 태스크 데이터
- **faiss/**: 벡터 검색을 위한 FAISS 인덱스 파일
- **parameter/**: 훈련된 프로젝터 가중치 파일

## 📋 요구사항

### 시스템 요구사항
- Python 3.8+
- CUDA 지원 GPU
- 16GB+ GPU 메모리 권장

### 주요 의존성
```bash
pip install torch pytorch-lightning transformers
pip install sentence-transformers faiss-cpu vllm
pip install wandb tqdm pytz
```

## 🛠️ 사용법

### 1. 모델 훈련
```bash
cd train_code
bash paraphrase_training.sh
```

### 2. 벡터 RAG 추론 (메인)
```bash
cd inference_code
bash run_inference.sh --dataset movie --confidence_threshold 0.4
```

### 3. Naive RAG 추론
```bash
cd inference_naive_RAG
bash run_inference.sh --dataset asqa --max_samples 10
```

### 4. No RAG 추론 (베이스라인)
```bash
cd inference_code_noRAG  
bash run_inference.sh --dataset sport --temperature 0.7
```

## ⚙️ 주요 매개변수

### 훈련 매개변수
- `--model_name`: 베이스 LLM 모델 (기본: DeepSeek-R1-Distill-Llama-8B)
- `--hidden_size`: 프로젝터 히든 사이즈 (기본: 4096)
- `--batch_size`: 배치 크기
- `--lr`: 학습률 (기본: 3e-5)
- `--max_epochs`: 최대 에포크 수

### 추론 매개변수
- `--confidence_threshold`: 확신도 임계값 (낮을수록 더 자주 RAG 수행)
- `--max_length`: 최대 생성 길이 (기본: 8192)
- `--temperature`: 생성 temperature (기본: 0)
- `--dataset`: 데이터셋 타입 (asqa, sport, movie, linguini)

## 🔍 핵심 기술적 특징

### 1. 적응적 RAG
- 모델의 확신도를 실시간 측정하여 필요할 때만 검색 수행
- 효율성과 성능의 균형 달성

### 2. 벡터 임베딩 주입
- `[CTX]` 특별 토큰을 통해 검색된 벡터를 LLM에 직접 주입
- 텍스트 변환 없이 벡터 정보를 효율적으로 활용

### 3. 메모리 최적화
- 베이스 LLM frozen, 프로젝터만 학습
- 그라디언트 체크포인팅으로 메모리 사용량 최소화
- bfloat16 정밀도로 연산 효율성 향상

### 4. 다양한 비교 모드
- 벡터 RAG vs Naive RAG vs No RAG 성능 비교 가능
- 확신도 임계값 조정을 통한 RAG 빈도 제어

## 📊 지원 데이터셋
- **ASQA**: Answer Sentence Question Answering
- **BBEH 영화 추천**: 영화 추천 대화 데이터
- **BBEH 스포츠 QA**: 스포츠 관련 질문 답변
- **BBEH Linguini**: 언어학적 질문 답변

## 📈 성능 및 로깅
- **Wandb 통합**: 훈련 과정 실시간 모니터링
- **상세 로깅**: 추론 과정, RAG 수행 횟수, 확신도 등 기록
- **결과 저장**: JSON 형태로 추론 결과 및 메타데이터 저장

## 🔧 개발 및 디버깅
- 디버깅 모드 지원
- 샘플 수 제한 기능으로 빠른 테스트 가능
- GPU 메모리 사용량 최적화
- 자동 패키지 설치 및 환경 설정

---

이 프로젝트는 벡터 기반 RAG 시스템의 효율적인 구현을 통해 질문 답변, 추천 시스템, 대화 생성 등 다양한 NLP 태스크에서 높은 성능을 달성하는 것을 목표로 합니다. 