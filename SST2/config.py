import torch

class Config:
    model_teacher = "textattack/bert-base-uncased-SST-2"
    student_num_layers = 6
    num_labels = 2
    batch_size = 32
    epochs = 5
    lr = 5e-5
    max_len = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    temperature = 2.0
    alpha = 0.5
    beta = 100.0
    teacher_layers = [2, 4, 6, 8, 10, 12]
    student_layers = [1, 2, 3, 4, 5, 6]
