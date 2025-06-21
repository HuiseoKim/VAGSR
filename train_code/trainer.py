import logging
import wandb
import numpy as np
import random
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from transformers import AutoTokenizer, AutoModelForCausalLM

from model import VectorRagModel
from data import data_pipeline

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

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'{config.checkpoint_path}',
        filename=f"{config.task}-{model_name}-{config.dataset_name}-batch_size_{config.batch_size * config.accumulate_grad_batches}-seed_{config.seed}-{config.lr}-{config.lr_scheduler}-{config.memo}"+"-{val_loss:.2f}-{epoch}epoch",
        save_top_k=6,
        save_last=False,
        verbose=True,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callback_list = [checkpoint_callback, lr_monitor] if config.early_stop != 0 else [lr_monitor, checkpoint_callback]

    # Compute effective batch size and set validation check interval to run validation every 100,000 examples
    effective_bs = config.batch_size * config.accumulate_grad_batches
    val_check_interval = max(1, 150000 // config.batch_size)

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