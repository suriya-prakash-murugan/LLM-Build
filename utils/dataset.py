import torch
import os
from pathlib import Path
from torch.utils.data import Dataset

class ValyrianDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, block_size=64):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y
