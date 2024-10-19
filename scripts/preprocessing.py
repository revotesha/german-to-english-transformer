import json
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def generate_square_subsequent_mask(sz):
    # I borrowed this code from the transformers library
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def read_corpus(filename):

    data = {"german": [], "english": []}
    with open(filename, "r", encoding="utf-8") as file:
        for line in file.readlines():
            json_line = json.loads(line)
            data["german"].append(json_line["de"] + " [EOS]")
            data["english"].append(
                "[BOS] " + json_line["en"] + " [EOS]"
            )  # shift target right
    return data


def tokenize(text, tokenizer, language="english"):

    return tokenizer(
        text[language],
        add_special_tokens=False,
        max_length=100,  # attn masking and padding masking is easier when fixed
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def get_tokenizers(
    german_model="dbmdz/bert-base-german-cased", english_model="bert-base-uncased"
):
    german_tokenizer = AutoTokenizer.from_pretrained(
        german_model, clean_up_tokenization_spaces=True
    )  # casing matters in German
    english_tokenizer = AutoTokenizer.from_pretrained(
        english_model, clean_up_tokenization_spaces=True
    )
    german_tokenizer.add_tokens(["[EOS]"], special_tokens=True)
    english_tokenizer.add_tokens(["[BOS]", "[EOS]"], special_tokens=True)

    german_vocab_size = german_tokenizer.vocab_size + 1  # count [EOS]
    english_vocab_size = english_tokenizer.vocab_size + 2  # count [BOS] & [EOS]

    return (german_tokenizer, english_tokenizer, german_vocab_size, english_vocab_size)


def get_data_loader(raw_data, german_tokenizer, english_tokenizer, batch_size=128):

    src_data = raw_data.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": german_tokenizer, "language": "german"},
    )
    trg_data = raw_data.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": english_tokenizer, "language": "english"},
    )

    del raw_data

    src_data = src_data.remove_columns(["german", "english", "token_type_ids"])
    trg_data = trg_data.remove_columns(["german", "english", "token_type_ids"])

    src_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    trg_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

    src_loader = DataLoader(src_data, batch_size=batch_size)
    trg_loader = DataLoader(trg_data, batch_size=batch_size)

    return src_loader, trg_loader
