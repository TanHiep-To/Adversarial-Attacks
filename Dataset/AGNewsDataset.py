import torch
from torch.utils.data import Dataset, DataLoader

class AGNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def get_text_and_label(self, idx):
        text = self.encodings.tokenizer.decode(self.encodings['input_ids'][idx])
        label = self.labels[idx]
        return text, label