# model.py
from typing import Dict, Union, Tuple, List
import torch
from torch import nn
import pytorch_lightning as pl
import logging
import os
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_with_hard_restarts_schedule_with_warmup, LambdaLR

class Projector(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(4096, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class VectorRagModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # projector (vector 변환)만 학습합니다.
        self.projector_ctx2llm = Projector(config.hidden_size)

        # 토크나이저에 특별 토큰 "[CTX]"를 추가합니다.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        special_tokens_dict = {"additional_special_tokens": ["[CTX]"]}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)

        # base model을 불러오고, 추가된 토큰이 있다면 임베딩 크기를 재조정합니다.
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,  # 메모리 사용량 감소
            low_cpu_mem_usage=True,      # CPU 메모리 사용 최소화
        )
        if num_added_tokens > 0:
            base_model.resize_token_embeddings(len(self.tokenizer))
        
        # 그라디언트 체크포인팅 활성화 (메모리 사용량 크게 감소)
        base_model.gradient_checkpointing_enable()
        base_model = base_model.to(self.device)

        # 슬라이딩 윈도우 어텐션 문제 해결: eager 모드에서는 해당 기능 비활성화
        if hasattr(base_model.config, "use_sliding_window_attention") and base_model.config.use_sliding_window_attention:
            logging.info("Disabling sliding window attention in eager mode.")
            base_model.config.use_sliding_window_attention = False

        # base model의 모든 파라미터 freeze
        for param in base_model.parameters():
            param.requires_grad = False
        self.model = base_model
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = 'left'
        self.tokenizer.add_eos_token = True

        # 추가: FAISS index와 metadata 로드 (memory augmentation)
        if config.memory_path:
            try:
                import faiss
                self.memory_index = faiss.read_index(config.memory_path)
                import pickle
                meta_path = os.path.join(os.path.dirname(config.memory_path), config.meta_path)
                with open(meta_path, "rb") as f:
                    self.memory_metadata = pickle.load(f)
                logging.info("Memory index and metadata loaded successfully.")
            except Exception as e:
                logging.warning("Failed to load memory index or metadata: " + str(e))
                self.memory_index = None
                self.memory_metadata = None
        else:
            self.memory_index = None
            self.memory_metadata = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> Union[Tuple, CausalLMOutput]:
        device = batch['input_ids'].device
        
        # [CTX] 토큰의 ID 찾기
        ctx_token_id = self.tokenizer.convert_tokens_to_ids("[CTX]")
        
        # 입력 텍스트 토큰 임베딩 얻기
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # [CTX] 토큰 위치 찾기 (배치 내 각 시퀀스별로)
        batch_size = input_ids.size(0)
        modified_input_embeds = []
        
        # gt_embedding 준비
        gt_embedding = batch['gt_embedding'].to(device)
        if gt_embedding.dim() == 4:
            gt_embedding = gt_embedding.squeeze(1)  # shape becomes [B, 1, H]
        # projected vector token 
        vector_token = self.projector_ctx2llm(gt_embedding)  # shape: [B, 1, H]
        
        # 각 배치 항목마다 [CTX] 토큰 위치 찾아서 대체
        embeddings = self.model.get_input_embeddings()
        for i in range(batch_size):
            # 해당 시퀀스의 입력 ID
            seq_ids = input_ids[i]
            # [CTX] 토큰 위치 찾기
            ctx_positions = (seq_ids == ctx_token_id).nonzero(as_tuple=True)[0]
            
            # 현재 시퀀스의 임베딩 얻기
            seq_embeds = embeddings(seq_ids)
            
            # [CTX] 토큰이 있는 경우
            if len(ctx_positions) > 0:
                ctx_pos = ctx_positions[0].item()  # 첫 번째 [CTX] 토큰 위치 사용
                # 해당 위치의 임베딩을 vector_token으로 대체
                seq_embeds[ctx_pos] = vector_token[i, 0]
            
            modified_input_embeds.append(seq_embeds)
        
        # 배치 내 모든 수정된 임베딩 결합
        input_embeds = torch.stack(modified_input_embeds)
        
        # labels 그대로 사용 (위치 조정 필요 없음)
        if 'labels' in batch and batch['labels'] is not None:
            labels = batch['labels'].to(device)
        else:
            labels = None
        
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        lm_loss = outputs.loss
        self.log('train_loss', lm_loss, prog_bar=True, sync_dist=True, logger=True)
        return lm_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self.forward(batch)
            loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, logger=True)
        return loss

    def on_validation_end(self):
        """검증 단계가 끝난 후 메모리를 정리합니다."""
        # 사용하지 않는 캐시 메모리 비우기
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, input_text: str) -> str:
        with torch.no_grad():
            tokenized = self.tokenizer(input_text, return_tensors="pt")
            input_ids = tokenized['input_ids'].to(self.device)
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=512,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def configure_optimizers(self) -> Tuple[List[AdamW], List[dict]]:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.projector_ctx2llm.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.projector_ctx2llm.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optim = AdamW(optimizer_grouped_parameters, lr=self.config.lr, eps=1e-8)
        total_steps = self.trainer.estimated_stepping_batches if hasattr(self, 'trainer') and self.trainer else 1000
        if self.config.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=int(total_steps * self.config.num_warmup_steps_ratio),
                num_training_steps=total_steps,
            )
        elif self.config.lr_scheduler == "constant":
            scheduler = get_constant_schedule(optim)
        elif self.config.lr_scheduler == "cosine":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optim,
                num_warmup_steps=int(total_steps * self.config.num_warmup_steps_ratio),
                num_training_steps=total_steps,
            )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optim], [scheduler]