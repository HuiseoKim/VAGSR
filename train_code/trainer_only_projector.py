import logging
import wandb
import numpy as np
import random
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger
import os

from transformers import AutoTokenizer, AutoModelForCausalLM

from model import VectorRagModel
from data import data_pipeline

class ProjectorCheckpoint(Callback):
    """
    프로젝터(projector)만 저장하는 커스텀 콜백
    전체 모델이 아닌 projector만 저장하여 메모리 사용을 최소화합니다.
    """
    def __init__(self, dirpath, monitor='val_loss', mode='min', save_top_k=3, filename=None):
        super().__init__()
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename
        
        self.best_k_models = {}
        self.kth_best_model = None
        self.kth_value = None
        
        if mode == 'min':
            self.compare = lambda x, y: x < y
            self.kth_value = float('inf')
        else:  # 'max'
            self.compare = lambda x, y: x > y
            self.kth_value = float('-inf')
            
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            
    def _get_metric_value(self, trainer):
        """현재 모니터링 중인 지표 값을 가져옵니다."""
        logger_logs = trainer.callback_metrics
        if self.monitor not in logger_logs:
            return None
        return logger_logs[self.monitor].item()
    
    def on_validation_end(self, trainer, pl_module):
        """검증 단계가 끝날 때 projector 체크포인트를 저장합니다."""
        # sanity check 단계에서는 체크포인트 저장하지 않음
        if trainer.sanity_checking:
            return
            
        if trainer.global_rank != 0:
            return  # 다중 GPU 환경에서는 메인 프로세스만 저장
        
        metric_value = self._get_metric_value(trainer)
        if metric_value is None:
            return
            
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step
        
        # 파일명 구성
        if self.filename is None:
            filename = f"projector-epoch{current_epoch}-step{global_step}-{self.monitor}={metric_value:.4f}.pth"
        else:
            # 사용자 지정 파일명에 지표와 에포크 정보 추가
            filename = f"{self.filename.format(epoch=current_epoch, step=global_step, val_loss=metric_value)}.pth"
            
        filepath = os.path.join(self.dirpath, filename)
        
        # 최상위 k개 모델 유지 로직
        if len(self.best_k_models) < self.save_top_k or self.compare(metric_value, self.kth_value):
            # projector 상태 저장
            self._save_projector(pl_module, filepath)
            
            # 저장된 모델 목록 업데이트
            self.best_k_models[filepath] = metric_value
            
            if len(self.best_k_models) > self.save_top_k:
                # 가장 성능이 낮은 모델 찾기
                worst_filepath = None
                for path, val in self.best_k_models.items():
                    if self.mode == 'min':
                        condition = worst_filepath is None or val > self.best_k_models[worst_filepath]
                    else:
                        condition = worst_filepath is None or val < self.best_k_models[worst_filepath]
                    if condition:
                        worst_filepath = path
                
                # 가장 성능이 낮은 모델 삭제
                if worst_filepath is not None:
                    if os.path.exists(worst_filepath):
                        os.remove(worst_filepath)
                    self.best_k_models.pop(worst_filepath)
                    
            # kth 값 업데이트
            if self.mode == 'min':
                self.kth_value = max(self.best_k_models.values())
            else:
                self.kth_value = min(self.best_k_models.values())
            
            # 로그
            trainer.logger.log_metrics({"projector_saved": 1.0})
            print(f"\nProjector 저장됨: {filepath} (성능: {metric_value:.4f})")
    
    def _save_projector(self, pl_module, filepath):
        """모델의 projector 부분만 저장합니다."""
        projector = pl_module.projector_ctx2llm
        torch.save(projector.state_dict(), filepath)

def train(config):
    devices = None
    accelerator = None
    if config.device == -1:
        accelerator = "cpu"
    else:
        accelerator = "gpu"
        temp = config.device.split(",")
        devices = [int(x) for x in temp]

    model_name = config.model_name.split("/")[-1]
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=f"{config.task}-{model_name}-{config.dataset_name}-batch_size {config.batch_size * config.accumulate_grad_batches}-{config.lr}-{config.lr_scheduler}-{config.memo}"
    )
    logging.info("-" * 30 + "Wandb 설정 완료!" + "-" * 30)

    pl.seed_everything(config.seed)
    logging.info(f"{'-' * 30} Seed {config.seed} 설정 완료! {'-' * 30}")

    model = VectorRagModel(config=config)
    logging.info("-" * 30 + "모델 초기화 완료!" + "-" * 30)

    tokenizer = model.tokenizer
    data_module = data_pipeline(config.train_data_path, config.valid_data_path, tokenizer, config)
    logging.info("-" * 30 + "데이터 로딩 완료!" + "-" * 30)

    # 일반 체크포인트 대신 projector 체크포인트 사용
    checkpoint_filename = f"{config.task}-{model_name}-{config.dataset_name}-batch_size_{config.batch_size * config.accumulate_grad_batches}-seed_{config.seed}-{config.lr}-{config.lr_scheduler}-{config.memo}"
    
    # projector만 저장하는 커스텀 체크포인트 콜백 생성
    projector_checkpoint = ProjectorCheckpoint(
        dirpath=f'{config.checkpoint_path}/projector',
        monitor='val_loss',
        save_top_k=3,
        mode="min",
        filename=checkpoint_filename+"-{val_loss:.2f}-{epoch}epoch"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callback_list = [projector_checkpoint, lr_monitor]

    # Compute effective batch size and set validation check interval to run validation every 100,000 examples
    effective_bs = config.batch_size * config.accumulate_grad_batches
    val_check_interval = max(1, 120000 // config.batch_size)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=config.precision,
        strategy=config.strategy,
        enable_progress_bar=True,
        callbacks=callback_list,
        max_epochs=config.max_epochs,
        num_sanity_val_steps=config.num_sanity_val_steps,
        logger=wandb_logger,
        accumulate_grad_batches=config.accumulate_grad_batches,
        val_check_interval=val_check_interval,
    )

    logging.info("-" * 30 + "학습 시작!" + "-" * 30)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=None)
    logging.info("-" * 30 + "학습 종료!" + "-" * 30)