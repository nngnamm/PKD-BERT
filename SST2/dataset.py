import torch
from torch.utils.data import Dataset

class SST2Dataset(Dataset):
    def __init__(self, dataset_list, tokenizer, max_len):
        self.dataset = dataset_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['sentence']
        label = item['label']

        # Sử dụng tokenizer(...) trực tiếp thay cho encode_plus
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
