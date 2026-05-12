import torch as t
from src import transformer, training, utils, data
from transformers import GPT2Tokenizer


def main(dataset_name, epochs):
    device = utils.get_device()

    cfg = transformer.Config(d_model = 64, n_ctx=64, d_head = 16, d_mlp = 256, n_heads = 4, n_layers = 4)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = transformer.Transformer(tokenizer, cfg).to(device)

    dataloader = data.load_dataloader(name=dataset_name, tokenizer=tokenizer)

    optimizer = t.optim.AdamW(model.parameters(), lr = 1e-3)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=utils.get_lr)


    for i in range(epochs):
        print(f"======== Epoch {i} ========")
        training.training_over_batch(model, dataloader, optimizer, scheduler)


if __name__ == "__main__":
    main(dataset_name="tinystories", epochs=5)
