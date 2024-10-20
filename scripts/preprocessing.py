"""
Module for data preprocessing, tokenization, and utility functions.

Author: Revo Tesha (https://www.linkedin.com/in/revo-tesha/)
"""

import json

from datasets import Dataset
from datasets.formatting.formatting import LazyBatch

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:

    # I borrowed this code from the transformers library
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def read_corpus(filename: str) -> dict[str, list[str]]:

    data = {"de": [], "en": []}
    with open(filename, "r", encoding="utf-8") as file:
        for line in file.readlines():
            json_line = json.loads(line)
            data["de"].append(json_line["de"] + " [EOS]")
            data["en"].append(
                "[BOS] " + json_line["en"] + " [EOS]"
            )  # shift target right
    return data


def tokenize(
    text: LazyBatch, tokenizer: BertTokenizerFast, language: str = "en"
) -> BatchEncoding:

    return tokenizer(
        text[language],
        add_special_tokens=False,
        max_length=100,  # attn masking and padding masking is easier when fixed
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def get_tokenizers(
    de_model: str = "dbmdz/bert-base-german-cased",
    en_model: str = "bert-base-uncased",
) -> tuple[BertTokenizerFast, BertTokenizerFast, int, int]:

    de_tokenizer = AutoTokenizer.from_pretrained(
        de_model, clean_up_tokenization_spaces=True
    )  # casing matters in German
    en_tokenizer = AutoTokenizer.from_pretrained(
        en_model, clean_up_tokenization_spaces=True
    )
    de_tokenizer.add_tokens(["[EOS]"], special_tokens=True)
    en_tokenizer.add_tokens(["[BOS]", "[EOS]"], special_tokens=True)

    de_vocab_size = de_tokenizer.vocab_size + 1  # count [EOS]
    en_vocab_size = en_tokenizer.vocab_size + 2  # count [BOS] & [EOS]

    return (de_tokenizer, en_tokenizer, de_vocab_size, en_vocab_size)


def get_data_loader(
    raw_data: Dataset,
    de_tokenizer: BertTokenizerFast,
    en_tokenizer: BertTokenizerFast,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader]:

    src_data = raw_data.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": de_tokenizer, "language": "de"},
    )
    trg_data = raw_data.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": en_tokenizer, "language": "en"},
    )

    del raw_data

    src_data = src_data.remove_columns(["de", "en", "token_type_ids"])
    trg_data = trg_data.remove_columns(["de", "en", "token_type_ids"])

    src_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    trg_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

    src_loader = DataLoader(src_data, batch_size=batch_size)
    trg_loader = DataLoader(trg_data, batch_size=batch_size)

    return src_loader, trg_loader
