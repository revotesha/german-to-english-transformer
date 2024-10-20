"""
Training script for model defined in model.py.

Author: Revo Tesha (https://www.linkedin.com/in/revo-tesha/)
"""

from datetime import datetime

from datasets import Dataset
from munch import Munch

import torch
import yaml

from model import Transformer
from scripts.preprocessing import (
    generate_square_subsequent_mask,
    get_data_loader,
    get_tokenizers,
    read_corpus,
)

# read configuration - update config.yaml as needed
with open("scripts/modeling/config.yaml") as file:
    config = Munch(yaml.safe_load(file))

print("\nPreparing training data...")
# data
train_data = Dataset.from_dict(
    read_corpus(f"{config.dataset['path']}/train_data.jsonl")
)
de_tokenizer, en_tokenizer, src_num_embeddings, trg_num_embeddings = get_tokenizers()
src_train_loader, trg_train_loader = get_data_loader(
    train_data, de_tokenizer, en_tokenizer
)

# model instantation
m_config = Munch(config.model)
model = Transformer(
    m_config.embed_dim,
    m_config.encoder_hidden_dim,
    m_config.decoder_hidden_dim,
    m_config.encoder_heads,
    m_config.decoder_heads,
    m_config.encoder_layers,
    m_config.decoder_layers,
    src_num_embeddings,
    trg_num_embeddings,
)

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# optimizer config
train_config = Munch(config.training)
criterion = torch.nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

# training
print("\nTraining...")
batches = len(src_train_loader)
for epoch in range(train_config.epochs):
    model.train()
    epoch_loss = 0
    for src_batch, trg_batch in zip(src_train_loader, trg_train_loader):
        src_x = src_batch["input_ids"].T.to(device)
        trg_x = trg_batch["input_ids"].T.to(device)
        labels = trg_batch["input_ids"][:, 1:]  # remove [BOS]

        labels = torch.cat(
            (labels, torch.zeros((labels.shape[0], 1))), dim=1
        )  # add a row of 0's to correct padding
        labels = labels.type(torch.LongTensor).to(device)

        trg_attn_mask = generate_square_subsequent_mask(trg_x.shape[0]).to(device)
        src_key_padding_mask = (src_batch["attention_mask"] == 0).to(device)
        trg_key_padding_mask = (trg_batch["attention_mask"] == 0).to(device)

        optimizer.zero_grad()

        output = model(
            src_x, trg_x, src_key_padding_mask, trg_key_padding_mask, trg_attn_mask
        )
        output = output.permute(1, 2, 0)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch: {epoch + 1}, Loss: {epoch_loss/batches}")

print("\nSaving model...")
# save trained model to 'models'
model_name = f"model_v{datetime.now().strftime('%m-%d-%Y@%H:%M:%S')}.pth"
torch.save(model.state_dict(), f"models/{model_name}")

# update config.yaml with latest model name for eval
config.training["model_name"] = model_name
with open("scripts/modeling/config.yaml", "w") as file:
    yaml.safe_dump(config, file, sort_keys=False)

print("Done! Model saved in the 'models' folder.")
