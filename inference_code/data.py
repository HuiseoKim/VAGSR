import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Dict, Optional, Any

class VectorRagDataset(Dataset):
    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        """
        벡터 증강 생성을 위한 데이터셋 로드 클래스
        다양한 형식의 데이터셋(asqa, BBEH_ 등)을 지원합니다.
        
        Args:
            data_path: 데이터 JSON 파일 경로
            max_samples: 로드할 최대 샘플 수 (None이면 전체 로드)
        """
        self.data_path = data_path
        self.max_samples = max_samples
        self.dataset_name = os.path.basename(data_path)
        self.data = self._load_data()
        logging.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """데이터 파일을 로드하고 처리합니다."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            processed_data = []
            
            # 데이터 형식 감지 및 처리
            if self.dataset_name.startswith('BBEH_'):
                # BBEH_ 형식: id, question, ground_truth만 포함하는 형식
                logging.info(f"Detected BBEH_ format dataset: {self.dataset_name}")
                
                # examples 키가 있는 경우 (원본 형식)
                if isinstance(data, dict) and 'examples' in data:
                    examples = data.get('examples', [])
                    for idx, ex in enumerate(examples, start=1):
                        q = ex.get('input', '').strip()
                        gt = ex.get('target', '').strip()
                        
                        item = {
                            "id": f"BBEH_{idx}",
                            "question": q,
                            "ground_truth": gt,
                            "input_text": self._prepare_input_text({"question": q})
                        }
                        processed_data.append(item)
                else:
                    # 이미 변환된 형식
                    for item in data:
                        processed_item = {
                            "id": item.get("id", ""),
                            "question": item.get("question", ""),
                            "ground_truth": item.get("ground_truth", ""),
                            "input_text": self._prepare_input_text({"question": item.get("question", "")})
                        }
                        processed_data.append(processed_item)
            
            elif self.dataset_name == 'asqa.json':
                # ASQA 형식: 특별한 형식 처리 필요
                logging.info("Detected ASQA format dataset")
                
                for item in data:
                    # instruction에서 질문 추출
                    instruction = item.get('instruction', '')
                    question_start = instruction.find('## Input:')
                    if question_start != -1:
                        question = instruction[question_start + len('## Input:'):].strip()
                    else:
                        question = instruction.strip()
                    
                    # 입력 형식 구성
                    processed_item = {
                        'id': item.get('id', ''),
                        'question': question,
                        'ground_truth': item.get('output', ''),
                        'input_text': self._prepare_input_text({"question": question})
                    }
                    processed_data.append(processed_item)
            
            else:
                # 기본 형식: 리스트 형태의 JSON 가정
                logging.info(f"Using default dataset format for {self.dataset_name}")
                
                for item in data:
                    processed_item = {
                        "id": item.get("id", ""),
                        "question": item.get("question", ""),
                        "ground_truth": item.get("ground_truth", ""),
                        "input_text": self._prepare_input_text({"question": item.get("question", "")})
                    }
                    processed_data.append(processed_item)
            
            # 최대 샘플 수 제한 적용
            if self.max_samples is not None and len(processed_data) > self.max_samples:
                processed_data = processed_data[:self.max_samples]
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error loading data from {self.data_path}: {e}")
            return []
    
    def _prepare_input_text(self, item: Dict) -> str:
        """
        데이터 항목에서 입력 텍스트를 구성합니다.
        데이터셋 유형에 따라 다른 형식을 적용할 수 있습니다.
        """
        # 기본 입력 형식 구성
        question = item['question']
        
        if self.dataset_name.startswith('BBEH_'):
            # BBEH_ 데이터셋의 입력 형식
            return f"Please analyze the question carefully and select the most appropriate answer from the given options. Your final answer should be in the format of a single option like (A), (B), (C), etc.\n\n{question}<answer>"
        else:
            # ASQA와 기타 데이터셋의 입력 형식
            return f"Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.\n\n{question}"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        """인덱스에 해당하는 데이터 항목을 반환합니다."""
        return self.data[idx]

def custom_collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """
    가변 길이 데이터를 처리하기 위한 커스텀 collate 함수
    
    각 배치 항목이 동일한 크기를 가질 필요가 없도록 합니다.
    대신 각 키마다 리스트 형태로 결과를 반환합니다.
    """
    result = {
        'id': [],
        'question': [],
        'ground_truth': [],
        'input_text': []
    }
    
    for item in batch:
        result['id'].append(item['id'])
        result['question'].append(item['question'])
        result['ground_truth'].append(item['ground_truth'])
        result['input_text'].append(item['input_text'])
    
    return result

def get_dataloader(config) -> DataLoader:
    """
    데이터셋의 DataLoader를 생성합니다.
    """
    dataset = VectorRagDataset(
        data_path=config.data_path,
        max_samples=config.max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,  # 배치 크기를 config에서 가져옴
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn  # 커스텀 collate 함수 사용
    )
    
    return dataloader 