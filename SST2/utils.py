import torch
import torch.nn as nn
import torch.nn.functional as F

def pkd_loss(student_logits, teacher_logits, labels, student_hidden, teacher_hidden, config):
    loss_ce = F.cross_entropy(student_logits, labels)

    student_probs = F.log_softmax(student_logits / config.temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / config.temperature, dim=-1)
    loss_kl = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (config.temperature ** 2)

    loss_pt = 0.0
    mse = nn.MSELoss()

    for t_idx, s_idx in zip(config.teacher_layers, config.student_layers):
        t_rep = teacher_hidden[t_idx][:, 0, :]
        s_rep = student_hidden[s_idx][:, 0, :]

        t_rep = F.normalize(t_rep, p=2, dim=1)
        s_rep = F.normalize(s_rep, p=2, dim=1)

        loss_pt += mse(s_rep, t_rep)

    loss_pt = loss_pt / len(config.student_layers)

    total_loss = (1.0 - config.alpha) * loss_ce + config.alpha * loss_kl + config.beta * loss_pt
    return total_loss
