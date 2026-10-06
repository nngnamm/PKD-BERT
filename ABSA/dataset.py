import torch
from torch.utils.data import Dataset

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

class ABSADataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len):
        texts = [item['text'] for item in raw_data]
        aspects = [item['span'] for item in raw_data]
        
        # Ánh xạ label chữ thành dạng số (0, 1, 2)
        labels = []
        for item in raw_data:
            lbl = item['label']
            if isinstance(lbl, str):
                labels.append(LABEL_MAP[lbl.lower()])
            else:
                labels.append(int(lbl))
        self.labels = labels

        print("-> Pre-tokenize dữ liệu dạng cặp (Text + Aspect)...")
        self.encodings = tokenizer(
            text=texts,
            text_pair=aspects,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings['token_type_ids'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
