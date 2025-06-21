from argparse import ArgumentParser
from trainer import train
import logging
import os

def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="None", type=str, help="huggingface model hub 상의 모델 이름")
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--device", default="-1", type=str)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--valid_batch_size", default=1, type=int) #validation 데이터 로드 시 사용
    parser.add_argument("--lr_scheduler", default="constant", type=str)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_warmup_steps_ratio", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float, help="weight decay")
    parser.add_argument("--max_epochs", default=20, type=int)
    parser.add_argument("--input_max_length", default=512, type=int, help="tokenizer 최대 입력 길이")
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--memory_path", default="None", type=str, help="FAISS index 파일 경로")
    parser.add_argument("--meta_path", default="None", type=str, help="FAISS metadata 파일 경로")
    parser.add_argument("--hidden_size", default=4096, type=int, help="projector의 hidden size")

    parser.add_argument("--memo", type=str, default="None") # wandb 메모
    parser.add_argument("--seed", default=31933, type=int)
    parser.add_argument("--checkpoint_path", default="./checkpoint", type=str)
    #parser.add_argument("--ckpt_path", default="None", type=str)
    parser.add_argument("--precision", default="bf16") ## bf16-mixed, bf16, 16-mixed, 16
    parser.add_argument("--early_stop", default=10, type=int)
    parser.add_argument("--num_sanity_val_steps", default=0, type=int)

    args = parser.parse_args()
    return args

def main(config):
    train(config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = parse_argument()
    main(config)