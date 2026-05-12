import torch as t
import math

def get_device():
    # Configure device
    if t.backends.mps.is_available():
        device = t.device('mps')
    elif t.cuda.is_available():
        device = t.device("cuda")
    else:
        device = t.device('cpu')
    print("Device: ", device)

def get_lr(step, warmup_steps=200, total_steps=10_000):
    """
    Get learning rate for linear warmup cosine annealing 
    """
    if step < warmup_steps:
        return step / warmup_steps  # linear warmup
    # cosine decay after warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
