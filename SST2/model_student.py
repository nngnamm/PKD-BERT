from transformers import BertConfig, BertForSequenceClassification

def get_student(config):
    student_config = BertConfig.from_pretrained(
        config.model_teacher,
        num_hidden_layers=config.student_num_layers,
        num_labels=config.num_labels,
        output_hidden_states=True
    )
    model = BertForSequenceClassification(student_config)
    return model
