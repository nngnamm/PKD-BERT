import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, f1_score

from config import Config
from dataset import SST2Dataset
from model_teacher import get_teacher
from model_student import get_student
from utils import pkd_loss

def evaluate(student, val_loader, config, accelerator):
    student.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = student(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            
            # Gather predictions across all GPUs for accurate evaluation
            preds = accelerator.gather(preds)
            labels = accelerator.gather(labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1

def train():
    config = Config()
    
    # 1. Initialize Accelerator for Multi-GPU (T4x2)
    accelerator = Accelerator()
    accelerator.print(f"=== Đang khởi chạy PKD trên Multi-GPU qua Accelerate ===")

    # 2. Load Tokenizer & Tải trực tiếp dữ liệu SST-2 dưới dạng Parquet
    tokenizer = AutoTokenizer.from_pretrained(config.model_teacher)

    try:
        train_url = "https://huggingface.co/datasets/glue/resolve/main/sst2/train-00000-of-00001.parquet"
        val_url = "https://huggingface.co/datasets/glue/resolve/main/sst2/validation-00000-of-00001.parquet"

        train_df = pd.read_parquet(train_url)
        val_df = pd.read_parquet(val_url)

        raw_train = train_df.to_dict(orient="records")
        raw_val = val_df.to_dict(orient="records")
    except Exception as e:
        sample_df = pd.read_csv("sample.csv")
        sample_df = sample_df.rename(columns={"text": "sentence"})
        raw_train = sample_df.to_dict(orient="records")
        raw_val = sample_df.to_dict(orient="records")

    train_dataset = SST2Dataset(raw_train, tokenizer, config.max_len)
    val_dataset = SST2Dataset(raw_val, tokenizer, config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 3. Khởi tạo Teacher & Student (Don't use .to(device) manually here)
    teacher = get_teacher(config)
    student = get_student(config)

    # 4. Cấu hình Optimizer & Scheduler
    optimizer = AdamW(student.parameters(), lr=config.lr, weight_decay=0.01)

    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 5. Prepare everything with Accelerator (Handles multi-GPU splitting automatically)
    student, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        student, optimizer, train_loader, val_loader, scheduler
    )
    teacher = accelerator.prepare(teacher)

    # 6. Huấn luyện PKD
    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(config.epochs):
        student.train()
        teacher.eval()
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", disable=not accelerator.is_local_main_process)

        for batch in progress_bar:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask=attention_mask)

            student_outputs = student(input_ids, attention_mask=attention_mask)

            loss = pkd_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs.logits,
                labels=labels,
                student_hidden=student_outputs.hidden_states,
                teacher_hidden=teacher_outputs.hidden_states,
                config=config
            )

            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Đánh giá sau mỗi epoch
        val_loss, val_acc, val_f1 = evaluate(student, val_loader, config, accelerator)
        
        if accelerator.is_local_main_process:
            print(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")
            print(f"Train Loss: {total_train_loss / len(train_loader):.4f}")
            print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc * 100:.2f}% | F1: {val_f1:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Unwrap model to save clean state_dict
                unwrapped_student = accelerator.unwrap_model(student)
                torch.save(unwrapped_student.state_dict(), "checkpoints/best_pkd_student_sst2.pt")
                print(f"--> [LƯU MODEL] Cập nhật checkpoint tốt nhất (Acc: {val_acc * 100:.2f}%)")
            print("-" * 60)

    accelerator.print("Hoàn tất huấn luyện!")

if __name__ == "__main__":
    train()
