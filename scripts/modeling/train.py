from datetime import datetime

from datasets import Dataset
import torch

from model import Transformer
from scripts.preprocessing import (
    generate_square_subsequent_mask,
    get_data_loader,
    get_tokenizers,
    read_corpus,
)

train_data = Dataset.from_dict(read_corpus("data/train_data.jsonl"))
german_tokenizer, english_tokenizer, src_num_embeddings, trg_num_embeddings = (
    get_tokenizers()
)
src_train_loader, trg_train_loader = get_data_loader(
    train_data, german_tokenizer, english_tokenizer
)

EMBED_DIM = 50
ENCODER_HIDDEN_DIM = 50
DECODER_HIDDEN_DIM = 50
ENCODER_HEADS = 1
DECODER_HEADS = 1
ENCODER_LAYERS = 1
DECODER_LAYERS = 1

model = Transformer(
    EMBED_DIM,
    ENCODER_HIDDEN_DIM,
    DECODER_HIDDEN_DIM,
    ENCODER_HEADS,
    DECODER_HEADS,
    ENCODER_LAYERS,
    DECODER_LAYERS,
    src_num_embeddings,
    trg_num_embeddings,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

EPOCHS = 1
criterion = torch.nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

batches = len(src_train_loader)

for epoch in range(EPOCHS):
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

model_path = f"models/model_v{datetime.now().strftime('%m-%d-%Y@%H:%M:%S')}.pth"
torch.save(model.state_dict(), model_path)
