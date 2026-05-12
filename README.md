# transformer-training
Implementation of a small language model, pre-training, post-training using DPO, interpretation of fine-tuning

# Transformer implementation
Here are some characteristics of my transformer implementation:
- use of LayerNorm
- d_mlp = 4*d_model ; d_model = n_heads * d_head
- MLP is composed of two projections, with a GeLU module in between
- Positional embedding is simply added to embedding

# Pre-training

I pre-trained the model over [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

I used an AdamW optimizer with Linear Warmup + Cosine Annealing scheduler.

Achieved a training loss of ~3 (cross-entropy).

# Next steps: 
## For pre-training
- RoPE embedding
- wandb logging
- Pre-train on distributed GPUs

## Supervised fine-tuning
- Simple pipeline for fine-tuning

## RLHF
- Implement DPO
- Identify alignment-related features
- Study their formation/evolution/deterioration at different steps of training
