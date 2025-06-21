#!/bin/bash

# 환경 설정
export WANDB_API_KEY="your_key"  # Wandb API 키 설정
export WANDB__SERVICE_WAIT="300"  # WANDB 연결 시간 최대 300초로 설정
export TOKENIZERS_PARALLELISM="false"  # 토크나이저 병렬 처리 방지

# 학습 설정
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TRAIN_DATA_PATH="/mnt/data1/paraphrase/data/train_filtered_400_700.parquet"
VALID_DATA_PATH="/mnt/data1/paraphrase/data/valid_modified.parquet"
DATASET_NAME="WIKI_dataset"
TASK="paraphrase"
WANDB_PROJECT="Vector-RAG"
DEVICE="0,1,2,3"
#STRATEGY="deepspeed_stage_3"
STRATEGY="ddp_find_unused_parameters_false"
BATCH_SIZE=1
LR=3e-5
LR_SCHEDULER="cosine"
NUM_WARMUP_STEPS_RATIO=0.05
WEIGHT_DECAY=0.2
MAX_EPOCHS=1
INPUT_MAX_LENGTH=736
ACCUMULATE_GRAD_BATCHES=8
#MEMORY_PATH="/work/VARSR/src/dataset/corpus_wiki/faiss/1_index.faiss"
#META_PATH="/work/VARSR/src/dataset/corpus_wiki/faiss/1_metadata.pkl"
MEMORY_PATH="None"
META_PATH="None"
HIDDEN_SIZE=4096

# Python 스크립트 실행
python3 ./train.py \
    --model_name $MODEL_NAME \
    --train_data_path $TRAIN_DATA_PATH \
    --valid_data_path $VALID_DATA_PATH \
    --dataset_name $DATASET_NAME \
    --task $TASK \
    --wandb_project $WANDB_PROJECT \
    --device $DEVICE \
    --strategy $STRATEGY \
    --batch_size $BATCH_SIZE \
    --lr_scheduler $LR_SCHEDULER \
    --lr $LR \
    --num_warmup_steps_ratio $NUM_WARMUP_STEPS_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --max_epochs $MAX_EPOCHS \
    --input_max_length $INPUT_MAX_LENGTH \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --memory_path $MEMORY_PATH \
    --meta_path $META_PATH \
    --hidden_size $HIDDEN_SIZE
