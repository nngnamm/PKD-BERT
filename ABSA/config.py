import torch

class Config:
    # Mô hình Teacher BERT & Dataset ABSA
    model_teacher = "bert-base-uncased"  # Lưu ý: Nên dùng checkpoint Teacher đã fine-tune trên ABSA để đạt F1 tối ưu
    dataset_name = "tomaarsen/setfit-absa-semeval-restaurants"
    num_labels = 3                       # 0: Negative, 1: Neutral, 2: Positive
    
    # Siêu tham số huấn luyện
    student_num_layers = 6
    batch_size = 16
    epochs = 5
    lr = 3e-5
    max_len = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tham số Patient Knowledge Distillation (PKD)
    temperature = 2.0
    alpha = 0.5                          # Trọng số Soft Logits Loss
    beta = 100.0                         # Trọng số Patient Hidden States Loss
    teacher_layers = [2, 4, 6, 8, 10, 12]
    student_layers = [1, 2, 3, 4, 5, 6]
