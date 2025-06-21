#!/bin/bash

# Vector Augmented Generation with Semantic Retrieval Inference 실행 스크립트

# 환경 변수 설정
export PYTHONPATH=/work/VAGSR_Inference:$PYTHONPATH

# 필요한 패키지 설치 확인
if ! python -c "import sentence_transformers" &> /dev/null; then
    echo "sentence-transformers 패키지를 설치합니다..."
    pip install -q sentence-transformers
    echo "sentence-transformers 패키지 설치 완료"
fi

if ! python -c "import vllm" &> /dev/null; then
    echo "vLLM 패키지를 설치합니다..."
    pip install -q vllm
    echo "vLLM 패키지 설치 완료"
fi

# 작업 디렉토리 설정
WORK_DIR="/work/VAGSR_Inference"
INFER_DIR="${WORK_DIR}/inference_naive_RAG"
DATA_DIR="${WORK_DIR}/data/task"

# 날짜와 시간을 이용한 실행 ID 생성
RUN_ID=$(date +"%Y%m%d_%H%M%S")
echo "Starting inference run with ID: ${RUN_ID}"

# 결과 디렉토리 생성
RESULTS_DIR="${INFER_DIR}/results/${RUN_ID}"
mkdir -p "${RESULTS_DIR}"

# 추론을 위한 기본 설정값
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # 기본 LLM 모델 이름
DATA_PATH="${DATA_DIR}/BBEH_movie_recommendation.json"  # 입력 데이터 경로
DATASET_TYPE="movie"  # 데이터셋 유형 (asqa, sport, movie, linguini)
FAISS_PATH="${WORK_DIR}/data/faiss"  # FAISS 인덱스 디렉토리
PARAMETER_PATH="${WORK_DIR}/data/parameter"  # 모델 가중치 디렉토리
PROJECTOR_FILENAME="projector_ctx2llm_1.10.pth"  # 프로젝터 가중치 파일 이름
MAX_LENGTH=8192  # 최대 생성 길이
#MAX_SAMPLES=10  # 처리할 최대 샘플 수 (없으면 전체 데이터셋 처리) - 디버깅 모드에서는 기본 5개로 설정
TEMPERATURE=0  # 생성 시 temperature 값
CONFIDENCE_THRESHOLD=0.4  # 확신도 임계값

# 디버깅 관련 설정
DEBUG_MODE=1  # 디버깅 모드 활성화 (1: 활성화, 0: 비활성화)

# GPU 관련 설정 - 단일 GPU만 사용하도록 설정
USE_MULTI_GPU=0  # 여러 GPU 사용 (1: 활성화, 0: 비활성화)
NUM_GPUS=1       # 사용할 GPU 개수 (단일 GPU 모드로 고정)
GPUS="0"         # 0번 GPU만 사용

# 명령행 인자 파싱
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --dataset)
      DATASET_TYPE="$2"
      # 데이터셋 유형에 따라 기본 경로 설정
      case "$DATASET_TYPE" in
        asqa)
          DATA_PATH="${DATA_DIR}/asqa.json"
          ;;
        sport)
          DATA_PATH="${DATA_DIR}/BBEH_sportQA.json"
          ;;
        movie)
          DATA_PATH="${DATA_DIR}/BBEH_movie_recommendation.json"
          ;;
        linguini)
          DATA_PATH="${DATA_DIR}/BBEH_linguini.json"
          ;;
        *)
          echo "Unknown dataset type: $DATASET_TYPE. Using custom data path."
          ;;
      esac
      shift 2
      ;;
    --faiss_path)
      FAISS_PATH="$2"
      shift 2
      ;;
    --parameter_path)
      PARAMETER_PATH="$2"
      shift 2
      ;;
    --projector_filename)
      PROJECTOR_FILENAME="$2"
      shift 2
      ;;
    --projector)
      # 프로젝터 버전 간편 선택 (예: --projector 1.10)
      PROJECTOR_VERSION="$2"
      PROJECTOR_FILENAME="projector_ctx2llm_${PROJECTOR_VERSION}.pth"
      shift 2
      ;;
    --max_length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --confidence_threshold)
      CONFIDENCE_THRESHOLD="$2"
      shift 2
      ;;
    --gpus)
      # 단일 GPU 모드에서는 무시하고 항상 0번 GPU 사용
      echo "단일 GPU 모드 실행 중. 지정된 GPU ID는 무시되고 GPU 0만 사용됩니다."
      GPUS="0"
      NUM_GPUS=1
      shift 2
      ;;
    --num_gpus)
      # 단일 GPU 모드에서는 무시
      echo "단일 GPU 모드 실행 중. 지정된 GPU 개수는 무시되고 1개만 사용됩니다."
      NUM_GPUS=1
      GPUS="0"
      shift 2
      ;;
    --use_multi_gpu)
      # 단일 GPU 모드에서는 무시
      echo "단일 GPU 모드로 강제 설정됨. 멀티 GPU 설정은 무시됩니다."
      USE_MULTI_GPU=0
      shift 2
      ;;
    --debug)
      DEBUG_MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 환경 설정 출력
echo "============= Inference Configuration ============="
echo "Model name: ${MODEL_NAME}"
echo "Dataset type: ${DATASET_TYPE}"
echo "Data path: ${DATA_PATH}"
echo "FAISS path: ${FAISS_PATH}"
echo "Parameter path: ${PARAMETER_PATH}"
echo "Projector file: ${PROJECTOR_FILENAME}"
echo "Max length: ${MAX_LENGTH}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Temperature: ${TEMPERATURE}"
echo "Confidence threshold: ${CONFIDENCE_THRESHOLD}"
echo "Debug mode: ${DEBUG_MODE}"
echo "단일 GPU 모드: GPU ${GPUS} 사용"
echo "Results directory: ${RESULTS_DIR}"
echo "=================================================="

# 프로젝터 파일 존재 여부 확인
PROJECTOR_FULL_PATH="${PARAMETER_PATH}/${PROJECTOR_FILENAME}"
if [ ! -f "${PROJECTOR_FULL_PATH}" ]; then
  echo "경고: 프로젝터 파일이 존재하지 않습니다: ${PROJECTOR_FULL_PATH}"
  echo "사용 가능한 프로젝터 파일:"
  ls -la ${PARAMETER_PATH}/projector_ctx2llm_*.pth 2>/dev/null || echo "프로젝터 파일이 없습니다."
  read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "실행이 취소되었습니다."
    exit 1
  fi
fi

# 출력 로그 파일 설정
LOG_FILE="${RESULTS_DIR}/inference_log.txt"

# 실행 명령어 구성
CMD="python ${INFER_DIR}/inference.py \
  --model_name ${MODEL_NAME} \
  --data_path ${DATA_PATH} \
  --faiss_path ${FAISS_PATH} \
  --parameter_path ${PARAMETER_PATH} \
  --projector_filename ${PROJECTOR_FILENAME} \
  --max_length ${MAX_LENGTH} \
  --temperature ${TEMPERATURE} \
  --confidence_threshold ${CONFIDENCE_THRESHOLD} \
  --output_path ${RESULTS_DIR} \
  --gpu_ids ${GPUS}"

# 최대 샘플 수가 설정된 경우에만 해당 플래그 추가
if [ ! -z "${MAX_SAMPLES}" ]; then
  CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

# 실행 명령어 출력
echo "Running command:"
echo "${CMD}"

# 명령어 실행 및 로그 파일에 출력 저장
eval "${CMD}" | tee "${LOG_FILE}"

# 실행 완료 후 결과 요약
echo "============= Inference Complete ============="
echo "Results saved to: ${RESULTS_DIR}"
echo "Log file: ${LOG_FILE}"

# 생성된 출력 파일 찾기 및 표시
echo "Generated files:"
find "${RESULTS_DIR}" -type f | sort

# 디버깅 모드가 활성화된 경우, 로그 파일에서 RAG 관련 정보 추출
if [ "$DEBUG_MODE" -eq 1 ]; then
  echo ""
  echo "============= RAG 디버깅 정보 ==============" 
  echo "문장 종료 시 확신도 확인:"
  grep -n "문장 종료 확인: 평균 확신도" "${LOG_FILE}" | tail -n 20
  
  echo ""
  echo "RAG 수행 횟수:"
  grep -n "낮은 확신도 감지" "${LOG_FILE}" | wc -l
  
  echo ""
  echo "[CTX] 토큰 정보:"
  grep -n "\[CTX\] 토큰 위치:" "${LOG_FILE}" | tail -n 10
  
  echo ""
  echo "벡터 증강 정보:"
  grep -n "벡터 증강 성공!" "${LOG_FILE}" | tail -n 10
  
  echo ""
  echo "RAG 요약 정보:"
  grep -n "RAG #" "${LOG_FILE}"
  
  echo ""
  echo "전체 RAG 수행 횟수:"
  grep -n "생성 완료! 총" "${LOG_FILE}" | tail -n 5
  
  echo ""
  echo "참고: 더 자세한 정보는 로그 파일을 확인하세요:"
  echo "less ${LOG_FILE}"
fi

echo "To analyze the results, check the JSON files in the results directory."
echo "================================================" 