
from transformers import AutoModelForSequenceClassification

def get_teacher(config):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_teacher,
        num_labels=config.num_labels,
        output_hidden_states=True
    )
    for param in model.parameters():
        param.requires_grad = False
    return model
