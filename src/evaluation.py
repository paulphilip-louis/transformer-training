import torch as t
from torch import Tensor
from jaxtyping import Int

# A few utils functions for sampling that i am going to move elsewhere


def sample_argmax(model, input:Int[Tensor, "batch seq"])->Int[Tensor, "batch 1"]:
    """
    Samples from model by taking the index of the maximal logit
    """
    logits = model(input) # (batch seq d_vocab)
    return logits[:, -1].argmax(-1)
    
def generate(model, input, max_new_tokens = 10):
    for _ in range(max_new_tokens):
        new_token = sample_argmax(model, input).unsqueeze(-1)
        input = t.cat((input, new_token), dim=-1)
    return input

def test_output(model, prompt="The little girl had"):
    tokens = model.tokenizer.encode(prompt)
    device = next(model.parameters()).device
    tokens = t.Tensor(tokens).type(dtype=t.int64).unsqueeze(0).to(device)
    output = generate(model, tokens, max_new_tokens=15)
    answer = model.tokenizer.decode(output.tolist())
    print(answer)