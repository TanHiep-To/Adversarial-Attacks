# Custom dataset class for TextAttack
class SimpleTextDataset:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = []
        self.labels = []

        for item in dataset:
            text = self.tokenizer.decode(item['input_ids'])
            label = item['labels'].item()
            self.texts.append(text)
            self.labels.append(label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
    