import json
import torch


def generate_square_subsequent_mask(sz):
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
