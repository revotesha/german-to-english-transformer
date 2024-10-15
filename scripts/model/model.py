import math
import torch
from torch import nn, Tensor

from preprocessing.tokenizers import german_tokenizer, english_tokenizer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input tensor.

        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``

        Returns:
            Tensor, shape ``[seq_len, batch_size, embedding_dim]``

        Raises:
            IndexError: if the positional encoding cannot be added to the input tensor
        """
        return x + self.pe[: x.size(0)]


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, output_dim, num_heads):
        super().__init__()
        self.mh_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(embed_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_key_padding_mask, dropout=True):
        # Multihead self-attention
        self_attn_output, _ = self.mh_self_attn(
            query=x, key=x, value=x, key_padding_mask=src_key_padding_mask
        )
        # Add multihead output to input and normalize
        self_attn_output = self_attn_output + x
        self_attn_output = self.norm1(self_attn_output)

        # Feedforward
        ff_output = self.relu(self.linear1(self_attn_output))
        ff_output = self.norm2(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)

        # Add attention output to feed forward output and normalize
        output = ff_output + self_attn_output
        output = self.norm3(output)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, output_dim, num_heads):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.mmh_self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mh_self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x,
        encoder_output,
        trg_key_padding_mask,
        src_key_padding_mask,
        trg_attn_mask,
        dropout=True,
    ):
        # Masked multihead self-attention
        self_attn_output, _ = self.mmh_self_attn(
            query=x,
            key=x,
            value=x,
            is_causal=False,
            attn_mask=trg_attn_mask,
            key_padding_mask=trg_key_padding_mask,
        )

        # Add self-attention output to input and normalize
        self_attn_output = self_attn_output + x
        self_attn_output = self.norm1(self_attn_output)

        # Multihead encoder-decoder attention
        attn_output, _ = self.mh_self_attn(
            query=self_attn_output,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=src_key_padding_mask,
        )

        # Add encoder-decoder attn output to decoder self-attn ouput and normalize
        attn_output = attn_output + self_attn_output
        attn_output = self.norm1(attn_output)

        # Feedforward
        ff_output = self.relu(self.linear1(attn_output))
        ff_output = self.norm2(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)

        # Add x to output and normalize
        output = ff_output + attn_output
        output = self.norm3(output)

        return output


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        output_dim,
        encoder_heads,
        decoder_heads,
        encoder_layers,
        decoder_layers,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(
            embedding_dim=embed_dim,
            num_embeddings=german_tokenizer.vocab_size + 1,
            padding_idx=0,  # add [EOS] to vocab size
        )
        self.trg_embedding = nn.Embedding(
            embedding_dim=embed_dim,
            num_embeddings=english_tokenizer.vocab_size + 2,
            padding_idx=0,  # add [BOS] and [EOS] to vocab size
        )
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(embed_dim, output_dim, encoder_heads)
                for _ in range(encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(embed_dim, output_dim, decoder_heads)
                for _ in range(decoder_layers)
            ]
        )
        self.linear = nn.Linear(embed_dim, english_tokenizer.vocab_size + 2)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        src_x,
        trg_x,
        src_key_padding_mask,
        trg_key_padding_mask,
        trg_attn_mask=None,
        decoder_only=False,
        return_encoder_output=False,
    ):

        trg_embed = self.trg_embedding(trg_x)
        trg = self.pos_encoder(trg_embed)

        # pos_encoder already does the summation
        # src = src_embed + src_pos
        # trg = trg_embed + trg_pos

        if not decoder_only:
            src_embed = self.src_embedding(src_x)
            src = self.pos_encoder(src_embed)
            for layer in self.encoder:
                src = layer(src, src_key_padding_mask)
        else:
            src = src_x

        for layer in self.decoder:
            trg = layer(
                trg, src, trg_key_padding_mask, src_key_padding_mask, trg_attn_mask
            )

        output = self.linear(trg)

        if return_encoder_output:
            output = output, src

        return output
