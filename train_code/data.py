import os
from typing import List, Dict
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer
import argparse
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning import LightningDataModule
from copy import deepcopy
import random

TASK_COL_NAME = {
    "paraphrase": ("id", "context", "embedding", "context_modified"),
    "downstream": ("id", "context", "embedding", "context_modified")
}

def data_pipeline(train_data_path: str,
                  valid_data_path: str,
                  tokenizer: AutoTokenizer,
                  config: argparse.Namespace) -> LightningDataModule:
    _, text_col_name, gt_embedding_col_name, context_modified_col_name = TASK_COL_NAME[config.task]
    data_module = VectorRAGDataModule(
        config=config,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        tokenizer=tokenizer,
        text_col_name=text_col_name,
        gt_embedding_col_name=gt_embedding_col_name,
        context_modified_col_name=context_modified_col_name,
        input_max_length=config.input_max_length,
        output_max_length=config.input_max_length,
        batch_size=config.batch_size,
    )
    return data_module

class ParquetIterableDataset(IterableDataset):
    def __init__(self,
                 parquet_path: str,
                 tokenizer: AutoTokenizer,
                 text_col_name: str,
                 gt_embedding_col_name: str = None,
                 context_modified_col_name: str = None,
                 input_max_length: int = 64):
        self.parquet_file = pq.ParquetFile(parquet_path)
        self.tokenizer = tokenizer
        self.text_col_name = text_col_name
        self.gt_embedding_col_name = gt_embedding_col_name
        self.context_modified_col_name = context_modified_col_name
        self.input_max_length = input_max_length
        self.num_rows = self.parquet_file.metadata.num_rows

    def __len__(self):
        import torch
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.num_rows
        # Partition the dataset length among workers (ceiling division)
        return (self.num_rows + worker_info.num_workers - 1) // worker_info.num_workers

    def __iter__(self):
        import torch
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        
        count = 0
        for batch in self.parquet_file.iter_batches(batch_size=1):
            if count % num_workers != worker_id:
                count += 1
                continue

            df_batch = batch.to_pandas()
            # 원래 context 텍스트를 가져옴
            context_text = df_batch[self.text_col_name].iloc[0]
            context_modified_text = df_batch[self.context_modified_col_name].iloc[0]
            # 랜덤 prefix 선택: 사용자 인스트럭션 중에서 랜덤으로 선택
            prefix_options = [
                "Rephrase the following sentence to maintain every detail and nuance exactly as it is, without omitting or changing any information: [CTX]",
                "Transform the sentence provided below into a new version that preserves its complete meaning, nuances, and all details exactly as given: [CTX]",
                "Your task is to rewrite the following sentence so that every nuance, detail, and aspect of its meaning is retained precisely; do not remove or alter any information: [CTX]",
                "Convert the sentence below into a paraphrased form that reflects the original meaning and all of its subtle details perfectly, without any omissions or modifications: [CTX]",
                "Please restate the following sentence in different words while ensuring that all details, nuances, and the exact meaning remain unchanged: [CTX]",
                "Reword the sentence below so that it conveys the same meaning, including every detail and nuance, without any alteration or exclusion: [CTX]",
                "Rewrite the sentence provided so that every aspect of its meaning, along with all details and subtle nuances, is fully preserved; do not leave out or change any part of it: [CTX]",
                "Your assignment is to create a paraphrase of the following sentence that retains its entire meaning, including all nuances and details, without any loss or modification: [CTX]",
                "Please transform the given sentence into a new expression that exactly preserves all the original details, nuances, and meaning, ensuring nothing is altered: [CTX]",
                "Recast the following sentence in your own words while keeping every detail, nuance, and aspect of its meaning intact, do not omit or change any information: [CTX]"
            ]
            prefix = random.choice(prefix_options)
            # 최종 입력 텍스트: prefix와 context의 결합
            new_text = prefix + context_modified_text

            encoding = self.tokenizer(
                new_text,
                truncation=True,
                max_length=self.input_max_length,
                padding='max_length',
                return_tensors='pt'
            )
            # 기본 input_ids, attention_mask, labels 구성
            input_ids = encoding['input_ids'].clone()
            attention_mask = encoding['attention_mask'].clone()
            labels = encoding['input_ids'].clone()

            # 마스킹: prefix 영역(예: 토큰화된 prefix 부분)을 -100으로 설정
            prefix_encoding = self.tokenizer(prefix, return_tensors='pt')
            prefix_len = prefix_encoding['input_ids'].size(1)
            labels[:, :prefix_len] = -100

            item = {
                'input_ids': input_ids.squeeze(0),
                'attention_mask': attention_mask.squeeze(0),
                'labels': labels.squeeze(0),
            }
            if self.gt_embedding_col_name:
                embedding = torch.tensor(df_batch[self.gt_embedding_col_name].iloc[0])
                if len(embedding.shape) == 1:
                    embedding = embedding.unsqueeze(0).unsqueeze(0)
                item['gt_embedding'] = embedding
            yield item
            count += 1

class VectorRAGDataModule(LightningDataModule):
    def __init__(self,
                 config,
                 train_data_path: str,
                 valid_data_path: str,
                 tokenizer: AutoTokenizer,
                 text_col_name: str,
                 gt_embedding_col_name: str = None,
                 context_modified_col_name: str = None,
                 input_max_length: int = 64,
                 output_max_length: int = 128,
                 batch_size: int = 64):
        super().__init__()
        self.config = config
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.text_col_name = text_col_name
        self.gt_embedding_col_name = gt_embedding_col_name
        self.context_modified_col_name = context_modified_col_name
        self.tokenizer = deepcopy(tokenizer)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.add_eos_token = True
        self.input_max_length = input_max_length
        self.batch_size = batch_size
        self.train_size = None
        self.valid_size = None

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ParquetIterableDataset(
                self.train_data_path,
                self.tokenizer,
                self.text_col_name,
                self.gt_embedding_col_name,
                self.context_modified_col_name,
                self.input_max_length
            )
            self.valid_dataset = ParquetIterableDataset(
                self.valid_data_path,
                self.tokenizer,
                self.text_col_name,
                self.gt_embedding_col_name,
                self.context_modified_col_name,
                self.input_max_length
            )
            self.train_size = self.train_dataset.parquet_file.metadata.num_rows
            self.valid_size = self.valid_dataset.parquet_file.metadata.num_rows

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config.valid_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2
        )

    def num_training_steps(self) -> int:
        if self.train_size is None:
            return None
        return self.train_size // self.batch_size

    def num_val_steps(self) -> int:
        if self.valid_size is None:
            return None
        return self.valid_size // self.batch_size