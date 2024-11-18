import os
import time
import json
import torch
import logging
import numpy as np
from collections import defaultdict
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, filename='train.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GRAD_SCALE_UPPER_BOUND_MEAN = 1000
GRAD_SCALE_LOWER_BOUND_MEAN = 0.01
THRESHOLD_GRAD_STATS = 100

class CustomQADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=32):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                question = item['question']
                answer = item['answers'][0]
                self.data.append((question, answer))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        source = self.tokenizer.encode_plus(
            question, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )
        target = self.tokenizer.encode_plus(
            answer, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )
        return {
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze()
        }

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    model_name = "google/t5-xl-lm-adapt"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print('model_load')
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    local_rank = setup_ddp()
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model._set_static_graph()  # Treat computation graph as static (experimental)
    print('model_distributed')

    train_dataset = CustomQADataset(file_path="/scratch1/tzhang62/Personalized_RAG/data/convai2/train.jsonl", tokenizer=tokenizer)
    print(f"Dataset size: {len(train_dataset)}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    print('data_load')

    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    tb_writer = SummaryWriter(log_dir="logs")
    scale = 2.0
    grad_stats = defaultdict(list)

    # Gradient accumulation setup
    gradient_accumulation_steps = 16  # Number of steps to accumulate gradients
    accumulation_counter = 0  # Counter for accumulated steps

    for epoch in range(3):
        train_loss = 0.0
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            labels = batch['labels'].to(local_rank)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps  # Scale loss for accumulation
            train_loss += loss.item()

            # Scale loss and backpropagate
            scaled_loss = scale * loss
            scaled_loss.backward()
            accumulation_counter += 1

            # Gradient stats for scaling adjustment
            for param in model.parameters():
                if param.grad is not None:  # Check if the gradient exists
                    grad_stats["max"].append(param.grad.max().item())
                    grad_stats["mean"].append(param.grad.mean().item())
                else:
                    grad_stats["max"].append(0.0)
                    grad_stats["mean"].append(0.0)

            if len(grad_stats["max"]) >= THRESHOLD_GRAD_STATS:
                if np.mean(grad_stats["max"]) > GRAD_SCALE_UPPER_BOUND_MEAN:
                    scale /= 2
                elif np.mean(grad_stats["mean"]) < GRAD_SCALE_LOWER_BOUND_MEAN:
                    scale *= 2
                grad_stats.clear()

            # Perform optimizer step and reset gradients every `gradient_accumulation_steps`
            if accumulation_counter % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), scale * 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                accumulation_counter = 0  # Reset accumulation counter

                # Clear GPU cache
                torch.cuda.empty_cache()

            # Log metrics
            if step % 10 == 0:
                tb_writer.add_scalar("Loss/train", train_loss / (step + 1), step)
                tb_writer.add_scalar("Scale", scale, step)
                tb_writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], step)
                logger.info(f"Step {step}: Loss = {train_loss / (step + 1)}, Scale = {scale}, LR = {scheduler.get_last_lr()[0]}")

            # Checkpoint save
            if step % 100 == 0 and local_rank == 0:
                model.module.save_pretrained(f"checkpoint_new/step-{step}")
                tokenizer.save_pretrained(f"checkpoint_new/step-{step}")

            # Clear GPU cache between mini-batches (optional)
            torch.cuda.empty_cache()
    # Save the last checkpoint
    if local_rank == 0:  # Ensure this is only done on the main process in a distributed setup
        model.module.save_pretrained("checkpoint_new/last")
        tokenizer.save_pretrained("checkpoint_new/last")
        print("Last checkpoint saved at 'checkpoint_new/last'")

    tb_writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()