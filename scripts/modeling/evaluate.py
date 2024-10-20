"""
Script for evaluating model trained with train.py.

Author: Revo Tesha (https://www.linkedin.com/in/revo-tesha/)
"""

import json
import random

from datasets import Dataset
from ignite.metrics import Rouge
from munch import Munch

import torch
from tqdm import tqdm
import yaml

from model import Transformer
from scripts.preprocessing import get_data_loader, get_tokenizers, read_corpus

# read configuration - update config.yaml as needed
with open("scripts/modeling/config.yaml") as file:
    config = Munch(yaml.safe_load(file))

# data
print("\nPreparing evaluation data...")
valid_data = Dataset.from_dict(
    read_corpus(f"{config.dataset['path']}/valid_data.jsonl")
)

de_tokenizer, en_tokenizer, src_num_embeddings, trg_num_embeddings = get_tokenizers()
src_valid_loader, trg_valid_loader = get_data_loader(
    valid_data,
    de_tokenizer,
    en_tokenizer,
    batch_size=config.dataset["valid_batch_size"],
)

# load model
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

model_name = config.training["model_name"]
model.load_state_dict(torch.load(f"models/{model_name}"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

EOS = 30523  # '[EOS]' will be used to stop token generation
samples = []  # stores sample translations
rouge = Rouge(variants=config.eval["rouge_vars"], multiref="best")  # metric

print("\nEvaluating...")
# evaluation
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

    # randomly select samples for visual inspection
    sample_prob = 0.1
    if random.random() < sample_prob:
        reference = reference[0]
        candidate = [en_tokenizer.convert_ids_to_tokens(int(id)) for id in candidate]
        reference = [en_tokenizer.convert_ids_to_tokens(int(id)) for id in reference]
        samples.append(
            {"candidate": " ".join(candidate), "reference": " ".join(reference)}
        )

print("\nSaving results...")
# compute rouge and save results to the 'results' folder
results = rouge.compute()
with open(f"results/rscores_{model_name.strip('.pth')}.txt", "w") as f:
    for metric, score in results.items():
        f.write(f"{metric}: {score}\n")

# save samples to the 'results' folder
with open(f"results/samples_{model_name.strip('.pth')}.jsonl", "w") as f:
    for item in samples:
        f.write(json.dumps(item) + "\n")
print("Done! See results in the 'results' folder.")
