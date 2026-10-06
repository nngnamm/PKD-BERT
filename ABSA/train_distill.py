import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from config import Config
from dataset import ABSADataset
from model_teacher import get_teacher
from model_student import get_student
from utils import pkd_loss

def evaluate(student, val_loader, config):
    student.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            token_type_ids = batch['token_type_ids'].to(config.device)
            labels = batch['labels'].to(config.device)

            outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs.logits

            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1

def train():
    config = Config()
    print(f"=== Đang khởi chạy PKD cho ABSA trên thiết bị: {config.device.upper()} ===")

    print("1/5. Tải tập dữ liệu ABSA SemEval Restaurants từ Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_teacher)
    
    dataset = load_dataset(config.dataset_name)
    raw_train = dataset['train']
    raw_val = dataset['test'] if 'test' in dataset else dataset['validation']

    train_dataset = ABSADataset(raw_train, tokenizer, config.max_len)
    val_dataset = ABSADataset(raw_val, tokenizer, config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print("2/5. Khởi tạo Teacher và Student...")
    teacher = get_teacher(config).to(config.device)
    student = get_student(config).to(config.device)

    optimizer = AdamW(student.parameters(), lr=config.lr, weight_decay=0.01)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler('cuda') if config.device == 'cuda' else None

    print("3/5. Bắt đầu quá trình PKD cho ABSA...")
    best_val_f1 = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(config.epochs):
        student.train()
        teacher.eval()
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            token_type_ids = batch['token_type_ids'].to(config.device)
            labels = batch['labels'].to(config.device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(config.device == 'cuda')):
                with torch.no_grad():
                    teacher_outputs = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )

                student_outputs = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                loss = pkd_loss(
                    student_logits=student_outputs.logits,
                    teacher_logits=teacher_outputs.logits,
                    labels=labels,
                    student_hidden=student_outputs.hidden_states,
                    teacher_hidden=teacher_outputs.hidden_states,
                    config=config
                )

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        val_loss, val_acc, val_f1 = evaluate(student, val_loader, config)
        print(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")
        print(f"Train Loss: {total_train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc * 100:.2f}% | Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(student.state_dict(), "checkpoints/best_pkd_student_absa.pt")
            print(f"--> [LƯU MODEL] Cập nhật checkpoint tốt nhất (Macro F1: {val_f1:.4f})")
        print("-" * 60)

    print("Hoàn tất huấn luyện ABSA!")

if __name__ == "__main__":
    train()
