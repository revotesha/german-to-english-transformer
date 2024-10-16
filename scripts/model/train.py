import os
from datasets import Dataset
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import Transformer
from utils import generate_square_subsequent_mask, read_corpus, tokenize

german_tokenizer = AutoTokenizer.from_pretrained(
    "dbmdz/bert-base-german-cased", clean_up_tokenization_spaces=True
)  # casing matters in German
english_tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased", clean_up_tokenization_spaces=True
)

german_tokenizer.add_tokens(["[EOS]"], special_tokens=True)
english_tokenizer.add_tokens(["[BOS]", "[EOS]"], special_tokens=True)

src_num_embeddings = german_tokenizer.vocab_size + 1  # add [EOS]
trg_num_embeddings = english_tokenizer.vocab_size + 2  # add [BOS] & [EOS]

EMBED_DIM = 50
ENCODER_HIDDEN_DIM = 50
DECODER_HIDDEN_DIM = 50
ENCODER_HEADS = 1
DECODER_HEADS = 1
ENCODER_LAYERS = 1
DECODER_LAYERS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
model = model.to(device)

EPOCHS = 1
criterion = torch.nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

if __name__ == "__main__":
    train_data = Dataset.from_dict(read_corpus("data/train_data.jsonl"))

    src_train_data = train_data.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": german_tokenizer, "language": "german"},
    )
    trg_train_data = train_data.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": english_tokenizer, "language": "english"},
    )

    del train_data

    src_train_data = src_train_data.remove_columns(
        ["german", "english", "token_type_ids"]
    )
    trg_train_data = trg_train_data.remove_columns(
        ["german", "english", "token_type_ids"]
    )

    src_train_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    trg_train_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

    src_train_loader = DataLoader(src_train_data, batch_size=128)
    trg_train_loader = DataLoader(trg_train_data, batch_size=128)

    batches = len(src_train_loader)
    print("Training...")
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
