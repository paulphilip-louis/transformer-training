import torch as t
from torch.utils.data import IterableDataset, DataLoader
import datasets

class StreamingTokenDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, seq_len, eot_token="<|endoftext|>"):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eot_id = tokenizer.encode(eot_token)[0]
    
    def __iter__(self):
        buffer = []
        for example in self.dataset:
            # Add tokenized example + eot token
            tokens = self.tokenizer.encode(example["text"])
            buffer.extend(tokens)
            buffer.append(self.eot_id)
            
            # If we exceed seq_len, we yield chunks until reaching below seq_len again
            while len(buffer) >= self.seq_len+1:
                chunk = buffer[:self.seq_len+1]
                buffer = buffer[self.seq_len:] # slight overlap
                yield t.tensor(chunk, dtype=t.long)

def load_dataset(name:str, tokenizer, window_size=64)->StreamingTokenDataset:
    if name == "tinystories":
        dataset = datasets.load_dataset("roneneldan/TinyStories", streaming=True, split="train")
        return StreamingTokenDataset(dataset, tokenizer, seq_len=64)
    else:
        raise ValueError("Name should be one of the following values : ['tinystories']")

def load_dataloader(name:str, tokenizer, window_size=64, batch_size=32)->DataLoader:
    dataset = load_dataset(name, tokenizer, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)
