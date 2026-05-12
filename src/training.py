import torch as t
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import einops
from jaxtyping import Float, Int
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from src import evaluation

@dataclass
class TrainingArgs:
    batch_size:int = 4
    epochs:int = 10
    lr:float=1e-5
    window_size = 8
    stride = 8


# Loss
def loss(logits:Float[Tensor, "batch seq d_vocab"], tokens:Int[Tensor, "batch seq"]):
    return F.cross_entropy(logits, tokens)


def training_over_batch(model, dataloader, optimizer, scheduler=None):
    device = next(model.parameters()).device
    step = 0
    for batch in dataloader:
        X = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)

        logits = model(X)
        loss = F.cross_entropy(logits.transpose(-1, -2), y) # d_vocab must be in 2nd position for logits

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
        if step % 10 == 0:
            print(f"Step {step} | Loss = {loss.item():.3f}")
        if step % 50 == 0:
            evaluation.test_output(model)
        step += 1
