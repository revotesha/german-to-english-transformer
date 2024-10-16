from datasets import Dataset
from ignite.metrics import Rouge

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tqdm import tqdm

from model import Transformer
import random

from utils import read_corpus, tokenize

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

MODEL_NAME = "model_v10-16-2024@10:17:09.pth"
model.load_state_dict(torch.load(f"models/{MODEL_NAME}"))
model = model.to(device)

if __name__ == "__main__":
    EOS = 30523
    trans_samples = []
    rouge = Rouge(variants=["L", 3, 2], multiref="best")

    valid_data = Dataset.from_dict(read_corpus("data/valid_data.jsonl"))

    src_valid_data = valid_data.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": german_tokenizer, "language": "german"},
    )
    trg_valid_data = valid_data.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": english_tokenizer, "language": "english"},
    )

    del valid_data

    src_valid_data = src_valid_data.remove_columns(
        ["german", "english", "token_type_ids"]
    )
    trg_valid_data = trg_valid_data.remove_columns(
        ["german", "english", "token_type_ids"]
    )

    src_valid_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    trg_valid_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

    src_valid_loader = DataLoader(src_valid_data, batch_size=1)
    trg_valid_loader = DataLoader(trg_valid_data, batch_size=1)

    for src_batch, trg_batch in tqdm(
        zip(src_valid_loader, trg_valid_loader), total=len(src_valid_loader)
    ):
        src_x = src_batch["input_ids"].T.to(device)
        src_key_padding_mask = (src_batch["attention_mask"] == 0).to(device)
        trg_x = trg_batch["input_ids"][:, 0].unsqueeze(0).to(device)

        model.eval()
        output, src_x = model(
            src_x,
            trg_x,
            src_key_padding_mask,
            trg_key_padding_mask=None,
            return_encoder_output=True,
        )
        output_ids = output.argmax(-1)

        src_seq_length = src_batch["attention_mask"].shape[1]
        for idx in range(src_seq_length):
            trg_x = torch.cat((trg_x, output_ids[idx, :].unsqueeze(0)))

            output = model(
                src_x,
                trg_x,
                src_key_padding_mask,
                trg_key_padding_mask=None,
                decoder_only=True,
            )
            output_ids = output.argmax(-1)

            if output_ids.T.squeeze(0)[-1].item() == EOS or idx == src_seq_length - 1:
                candidate = (trg_x.T.squeeze(0)[1:]).tolist()
                break

        reference = trg_batch["input_ids"][:, 1:].squeeze()
        padding_index = torch.where(reference == EOS)[0].item()
        reference = (reference[:padding_index]).tolist()

        reference = [[str(r) for r in reference]]
        candidate = [str(id) for id in candidate]

        rouge.update(([candidate], [reference]))

        number = random.randint(0, 64)
        if number == 4:
            trans_samples.append((candidate, reference))

    results = rouge.compute()
    with open(f"results/results_{MODEL_NAME.strip('.pth')}.txt", "w") as f:
        for metric, score in results.items():
            f.write(f"{metric}: {score}\n")
