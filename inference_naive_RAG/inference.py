import os
# CUDA_VISIBLE_DEVICES 설정은 parse_args()에서 처리하므로 제거
import json
import torch
import logging
import argparse
from tqdm import tqdm
from typing import Dict, List
import time
import datetime
import pytz
import pytorch_lightning as pl
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist

from model_hs import TextRagInferenceModel
from data import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Text Augmented Generation with Semantic Retrieval Inference")
    
    # 모델 관련 인자
    parser.add_argument("--model_name", type=str, default="r1-distill-llama-8b", help="사용할 기본 모델 이름")
    parser.add_argument("--device", type=str, default="cuda", help="사용할 디바이스 (cuda, cpu)")
    parser.add_argument("--gpu_ids", type=str, default="0", help="사용할 GPU ID (쉼표로 구분, 예: 0,1,2,3)")
    parser.add_argument("--hidden_size", type=int, default=4096, help="프로젝터의 hidden size")
    parser.add_argument("--batch_size", type=int, default=1, help="추론 시 배치 크기 (멀티 GPU 사용 시 증가 권장)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 설정")
    
    # 파일 경로 관련 인자
    parser.add_argument("--data_path", type=str, default="/work/VAGSR_Inference/data/task/asqa.json", help="입력 데이터 경로")
    parser.add_argument("--faiss_path", type=str, default="/work/VAGSR_Inference/data/faiss", help="FAISS 인덱스 디렉토리 경로")
    parser.add_argument("--parameter_path", type=str, default="/work/VAGSR_Inference/data/parameter", help="모델 가중치 디렉토리 경로")
    parser.add_argument("--projector_filename", type=str, default="projector_ctx2llm_1.10.pth", help="프로젝터 가중치 파일 이름")
    parser.add_argument("--output_path", type=str, default="/work/VAGSR_Inference/inference_code/results", help="결과 저장 경로")
    
    # 생성 관련 인자
    parser.add_argument("--max_length", type=int, default=512, help="최대 생성 길이")
    parser.add_argument("--temperature", type=float, default=0.7, help="생성 시 temperature 값")
    parser.add_argument("--confidence_threshold", type=float, default=0.3, help="확신도 임계값 (이 값보다 낮으면 FAISS 검색 수행)")
    
    # 데이터 처리 관련 인자
    parser.add_argument("--max_samples", type=int, default=None, help="처리할 최대 샘플 수")
    parser.add_argument("--num_workers", type=int, default=4, help="데이터 로딩 시 사용할 워커 수")
    
    args = parser.parse_args()
    return args

def set_up_logging(args):
    """로깅 설정"""
    log_dir = os.path.join(args.output_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 한국 시간대(KST)로 설정
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.datetime.now(kst)
    time_str = now.strftime('%Y%m%d_%H%M%S')
    
    log_path = os.path.join(log_dir, f"inference_{time_str}.log")
    
    # 로깅 레벨을 INFO로 변경하여 디버깅 정보 표시
    logging.basicConfig(
        level=logging.INFO,  # ERROR에서 INFO로 변경
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",  # 파일명과 라인 번호 추가
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # 라이브러리 로깅 레벨은 여전히 ERROR로 유지
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    
    logging.info("========== 텍스트 증강 생성 디버깅 로그 시작 ==========")
    logging.info(f"로그 파일 경로: {log_path}")
    logging.info(f"실행 인자: {args}")
    
    return log_path

def save_results(results: List[Dict], args):
    """결과를 JSON 파일로 저장"""
    os.makedirs(args.output_path, exist_ok=True)
    
    # 한국 시간대(KST)로 설정
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.datetime.now(kst)
    time_str = now.strftime('%Y%m%d_%H%M%S')
    
    output_file = os.path.join(args.output_path, f"results_{time_str}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Results saved to {output_file}")

def set_seed(seed):
    """시드 설정 (PyTorch Lightning 사용)"""
    pl.seed_everything(seed, workers=True)
    logging.info(f"Random seed set to {seed}")

def main():
    args = parse_args()
    
    try:
        # 결과 저장 디렉토리 생성
        os.makedirs(args.output_path, exist_ok=True)
        
        # 로깅 설정
        log_path = set_up_logging(args)
        
        # 시드 설정
        set_seed(args.seed)
        
        # GPU 설정 - 단일 GPU만 사용하도록 설정
        if args.device == "cuda":
            # 무조건 첫 번째 GPU만 사용하도록 설정
            args.gpu_ids = "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
            logging.info(f"단일 GPU 모드: GPU {args.gpu_ids} 사용")
            
            # CUDA 사용 가능 여부 확인
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logging.info(f"사용 GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logging.warning("CUDA를 사용할 수 없어 CPU로 실행합니다.")
        else:
            device = torch.device(args.device)
            logging.info(f"CPU 모드로 실행")
        
        # 모델 초기화
        model = TextRagInferenceModel(args)
        
        # 데이터 로더 생성
        dataloader = get_dataloader(args)
        
        # 결과 저장 리스트
        results = []
        
        # 모든 데이터에 대해 추론 실행
        for i, batch in enumerate(tqdm(dataloader, desc="Inferencing")):
            batch_size = len(batch['id'])
            
            for j in range(batch_size):
                sample_id = batch['id'][j]
                question = batch['question'][j]
                ground_truth = batch['ground_truth'][j]
                
                # 개별 샘플에 대한 정보 기록
                logging.info(f"Processing sample {i*args.batch_size+j+1}: {sample_id}")
                logging.info(f"질문: {question}")
                
                # 입력 텍스트 가져오기 - 이미 데이터로더에서 처리됨
                input_text = batch['input_text'][j]
                
                # 추론 시작
                try:
                    logging.info(f"Starting inference for sample {sample_id}")
                    start_time = time.time()
                    
                    # RAG 정보를 추적하기 위해 model 객체의 rag_info 속성을 초기화
                    if hasattr(model, 'rag_info'):
                        model.rag_info = []
                    
                    generated_text = model.generate_with_text_augmentation(
                        input_text,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        confidence_threshold=args.confidence_threshold
                    )
                    inference_time = time.time() - start_time
                    logging.info(f"Inference completed in {inference_time:.2f} seconds")
                    
                    # RAG 정보 가져오기 (있는 경우)
                    rag_info = getattr(model, 'rag_info', [])
                    
                    # 결과 저장
                    result = {
                        "id": sample_id,
                        "question": question,
                        "generated_text": generated_text,
                        "ground_truth": ground_truth,
                        "inference_time_seconds": inference_time,
                        "rag_count": len(rag_info) if rag_info else 0,
                        "confidence_threshold": args.confidence_threshold
                    }
                    
                    # 자세한 RAG 정보가 있으면 결과에 추가
                    if hasattr(model, 'rag_info') and model.rag_info:
                        result["rag_details"] = model.rag_info
                    
                    results.append(result)
                    
                    # 샘플 결과 로깅
                    logging.info(f"샘플 {sample_id} 처리 완료. 최종 텍스트 길이: {len(generated_text)}, RAG 수행 횟수: {result['rag_count']}")
                    
                except Exception as e:
                    logging.error(f"Error processing sample {sample_id}: {e}", exc_info=True)
        
        # 결과 저장
        logging.info(f"Saving {len(results)} results...")
        save_results(results, args)
        
        # 최종 결과 요약
        total_rags = sum(r.get('rag_count', 0) for r in results)
        avg_rags = total_rags / len(results) if results else 0
        
        logging.info(f"추론 완료! 총 {len(results)}개 샘플, 총 {total_rags}회 RAG 수행 (평균: {avg_rags:.2f}회/샘플)")
        logging.info(f"로그 파일: {log_path}")
        logging.info(f"결과 저장 경로: {args.output_path}")
        
    except Exception as e:
        logging.error(f"Error during inference process: {e}", exc_info=True)
        logging.error("Inference failed")
    finally:
        # 프로그램 종료 전 process group 정리
        if dist.is_initialized():
            dist.destroy_process_group()
            logging.info("Process group destroyed")

if __name__ == "__main__":
    main() 