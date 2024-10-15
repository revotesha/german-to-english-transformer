import torch

from model import Transformer
import random

from ignite.metrics import Rouge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_DIM = 200
OUTPUT_DIM = 200
ENCODER_HEADS = 4
DECODER_HEADS = 4
ENCODER_LAYERS = 1
DECODER_LAYERS = 1

model = Transformer(
    EMBED_DIM, OUTPUT_DIM, ENCODER_HEADS, DECODER_HEADS, ENCODER_LAYERS, DECODER_LAYERS
)

MODEL_PATH = ""
model.load_state_dict(torch.load(MODEL_PATH))
# model.load_state_dict(torch.load('model.pth', map_location=torch.device(device)))
model = model.to(device)

if __name__ == "__main__":
    EOS = 30523
    trans_samples = []
    rouge = Rouge(variants=["L", 3, 2], multiref="best")

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
        batch_size = trg_batch["attention_mask"].shape[1]
        for idx in range(batch_size):
            trg_x = torch.cat((trg_x, output_ids[idx, :].unsqueeze(0)))

            output = model(
                src_x,
                trg_x,
                src_key_padding_mask,
                trg_key_padding_mask=None,
                decoder_only=True,
            )
            output_ids = output.argmax(-1)

            if output_ids.T.squeeze(0)[-1].item() == EOS:
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

    rouge.compute()
