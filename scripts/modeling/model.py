"""
Module with transformer model classes.

Author: Revo Tesha (https://www.linkedin.com/in/revo-tesha/)
"""

import math
import torch

from torch import nn, Tensor
from typing import Union

# <!> PLEASE NOTE that docstrings were generated by ChatGPT and may contain mistakes <!>


class PositionalEncoding(nn.Module):
    """
    Class for adding positional encoding to the input tensor.
    """

    # Note: I borrowed this code from the spotpython library
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the model (the size of the embeddings).
            max_len (int, optional): The maximum length of the input sequence. Defaults to 5000.
        """

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
        Adds the positional encodings to the input tensor `x`.

        Args:
            x (Tensor): The input tensor with shape
                ``[seq_len, batch_size, embedding_dim]``.

        Returns:
            Tensor: The input tensor with positional encodings added, of shape
                ``[seq_len, batch_size, embedding_dim]``.

        Raises:
            IndexError: If the sequence length of the input tensor exceeds the
                maximum length set during initialization.
        """

        return x + self.pe[: x.size(0)]


class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder.

    This layer includes multi-head self-attention followed by a feedforward neural network.
    Both components have residual connections, normalization, and optional dropout.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int) -> None:
        """
        Initializes the EncoderLayer.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            hidden_dim (int): Dimensionality of the feedforward neural network.
            num_heads (int): Number of attention heads in the multi-head attention mechanism.
        """

        super().__init__()
        self.mh_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self, x: Tensor, src_key_padding_mask: Tensor, dropout: bool = True
    ) -> Tensor:
        """
        Forward pass of the encoder layer.

        Args:
            x (Tensor): The input tensor of shape ``[seq_len, batch_size, embed_dim]``.
            src_key_padding_mask (Tensor): A mask tensor indicating which positions should
                be ignored during attention.
            dropout (bool, optional): Whether to apply dropout in the feedforward network.
                Defaults to True.

        Returns:
            Tensor: The output tensor of shape ``[seq_len, batch_size, embed_dim]``, after
                attention and feedforward steps.
        """

        # Multihead self-attention
        self_attn_output, _ = self.mh_self_attn(
            query=x, key=x, value=x, key_padding_mask=src_key_padding_mask
        )
        # Add multihead output to input and normalize
        self_attn_output = self_attn_output + x
        self_attn_output = self.norm1(self_attn_output)

        # Feedforward
        ff_output = self.linear1(self_attn_output)
        ff_output = self.relu(ff_output)
        ff_output = self.norm2(ff_output)
        if dropout:
            ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)

        # Add attention output to feed forward output and normalize
        output = ff_output + self_attn_output
        output = self.norm3(output)

        return output


class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer decoder.

    This layer includes masked multi-head self-attention, multi-head encoder-decoder attention,
    and a feedforward neural network, all with residual connections and layer normalization.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int) -> None:
        """
        Initializes the DecoderLayer.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            hidden_dim (int): Dimensionality of the feedforward neural network.
            num_heads (int): Number of attention heads in the multi-head attention mechanism.
        """

        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.mmh_self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mh_self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        trg_key_padding_mask: Tensor,
        src_key_padding_mask: Tensor,
        trg_attn_mask: Tensor,
        dropout: bool = True,
    ) -> Tensor:
        """
        Forward pass of the decoder layer.

        Args:
            x (Tensor): The input tensor of shape ``[seq_len, batch_size, embed_dim]``.
            encoder_output (Tensor): The output tensor from the encoder, shape ``[seq_len, batch_size, embed_dim]``.
            trg_key_padding_mask (Tensor): A mask tensor indicating which positions in the target sequence should be ignored.
            src_key_padding_mask (Tensor): A mask tensor indicating which positions in the source sequence should be ignored.
            trg_attn_mask (Tensor): A mask tensor for preventing attention to future tokens in the target sequence.
            dropout (bool, optional): Whether to apply dropout in the feedforward network. Defaults to True.

        Returns:
            Tensor: The output tensor of shape ``[seq_len, batch_size, embed_dim]``, after attention and feedforward steps.
        """

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
        attn_output = self.norm2(attn_output)

        # Feedforward
        ff_output = self.linear1(attn_output)
        ff_output = self.relu(ff_output)
        ff_output = self.norm3(ff_output)
        if dropout:
            ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)

        # Add x to output and normalize
        output = ff_output + attn_output
        output = self.norm4(output)

        return output


class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.

    This model implements the Transformer architecture, which consists of an encoder and a decoder.
    It includes embedding layers for source and target inputs, positional encoding, and multiple
    encoder and decoder layers.
    """

    def __init__(
        self,
        embed_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_heads: int,
        decoder_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        src_num_embeddings: int,
        trg_num_embeddings: int,
    ) -> None:
        """
        Initializes the Transformer model.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            encoder_hidden_dim (int): Dimensionality of the hidden layers in the encoder.
            decoder_hidden_dim (int): Dimensionality of the hidden layers in the decoder.
            encoder_heads (int): Number of attention heads in the encoder.
            decoder_heads (int): Number of attention heads in the decoder.
            encoder_layers (int): Number of layers in the encoder.
            decoder_layers (int): Number of layers in the decoder.
            src_num_embeddings (int): Size of the source vocabulary.
            trg_num_embeddings (int): Size of the target vocabulary.
        """

        super().__init__()
        self.src_embedding = nn.Embedding(
            embedding_dim=embed_dim, num_embeddings=src_num_embeddings, padding_idx=0
        )
        self.trg_embedding = nn.Embedding(
            embedding_dim=embed_dim, num_embeddings=trg_num_embeddings, padding_idx=0
        )
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(embed_dim, encoder_hidden_dim, encoder_heads)
                for _ in range(encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(embed_dim, decoder_hidden_dim, decoder_heads)
                for _ in range(decoder_layers)
            ]
        )
        self.linear = nn.Linear(embed_dim, trg_num_embeddings)

    def forward(
        self,
        src_x: Tensor,
        trg_x: Tensor,
        src_key_padding_mask: Tensor,
        trg_key_padding_mask: Tensor,
        trg_attn_mask: Union[Tensor, None] = None,
        decoder_only: bool = False,
        return_encoder_output: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        Forward pass of the Transformer model.

        Args:
            src_x (Tensor): The source input tensor of shape ``[seq_len, batch_size]``.
            trg_x (Tensor): The target input tensor of shape ``[seq_len, batch_size]``.
            src_key_padding_mask (Tensor): A mask tensor for padding in the source input.
            trg_key_padding_mask (Tensor): A mask tensor for padding in the target input.
            trg_attn_mask (Union[Tensor, None], optional): A mask tensor for attention in the
                target input. Defaults to None.
            decoder_only (bool, optional): If True, only the decoder will be run. Defaults to False.
            return_encoder_output (bool, optional): If True, the output will include the encoder's
                output. Defaults to False.

        Returns:
            Union[Tensor, tuple[Tensor, Tensor]]: The output tensor of shape
                ``[seq_len, batch_size, trg_num_embeddings]`` or a tuple of (output, encoder output) if
                `return_encoder_output` is True.
        """

        trg_embed = self.trg_embedding(trg_x)
        trg = self.pos_encoder(trg_embed)

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
