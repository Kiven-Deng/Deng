
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import os
import datetime
from collections import OrderedDict
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple, Union

from .mallows import MallowsPhiSampler


###################
# Network definition
# Notation :
#            bsz : batch size
#            numb_items : number of candidates
#            dim_emb : embedding/hidden dimension
#            numb_heads : number of attention heads
#            dim_ff : feed-forward dimension
#            numb_layers : number of encoder/decoder layers
###################
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class EmbeddingType(Enum):
    """Enumeration for the different types of embedding layers available."""
    LINEAR = 'linear'
    CONV = 'conv'
    MALLOWS_IDENTITY = 'mallows_identity'

class AbstractEmbedding(nn.Module, ABC):
    """
    Abstract base class for all embedding layers.
    Ensures that any embedding module implements a forward pass and has an embedding_dim attribute.
    """
    def __init__(self):
        super().__init__()
        self.embedding_dim = 0

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes the raw input tensor and returns its embedding.

        Args:
            x (torch.Tensor): The input data. Shape will vary based on the data source.

        Returns:
            torch.Tensor: The embedded data. Shape: (batch_size, seq_len, embedding_dim).
        """
        pass

class LinearEmbedding(AbstractEmbedding):
    """
    A simple linear projection embedding layer, suitable for continuous feature inputs.

    Args:
        input_dim (int): The size of the input feature dimension (e.g., max_voters).
        embedding_dim (int): The desired size of the output embedding.
    """
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding_layer = nn.Linear(input_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the input features to the embedding dimension.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Embedded tensor. Shape: (batch_size, seq_len, embedding_dim).
        """
        return self.embedding_layer(x)

class ConvEmbedding(AbstractEmbedding):
    """
    An embedding layer using 1D convolutions that is robust to a variable number of input features (voters).
    This is useful for capturing local patterns in the voter rankings for each candidate.

    Args:
        embedding_dim (int): The desired size of the output embedding.
        out_channels (int): The number of output channels for the first convolutional layer.
    """
    def __init__(self, embedding_dim: int, out_channels: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Instead of AdaptiveAvgPool1d, we'll do masked average manually.
        self.linear = nn.Linear(out_channels * 2, embedding_dim)

    def forward(self, x: torch.Tensor, voter_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Passes the input through the convolutional network to get the embedding.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, variable_input_dim).

        Returns:
            torch.Tensor: Embedded tensor. Shape: (batch_size, seq_len, embedding_dim).
        """
        batch_size, seq_len, input_dim = x.shape
        x_reshaped = x.view(batch_size * seq_len, 1, input_dim)

        if voter_mask is not None:
             if voter_mask.dim() == 2:
                 voter_mask_expanded = voter_mask.unsqueeze(1).expand(batch_size, seq_len, input_dim)
             else:
                 voter_mask_expanded = voter_mask
             voter_mask_batch = voter_mask_expanded.sum(dim=2) # Shape: (batch_size, seq_len)

             # Process each item in the batch independently to avoid padding
             pooled_outs = []
             for i in range(batch_size):
                 # Find the number of valid voters for this item in the batch
                 # All tokens in the same batch item have the same number of valid voters
                 valid_len = voter_mask[i].sum().long().item()
                 # If valid_len is 0 (e.g., padded candidate), just use length 1 to avoid empty tensors
                 if valid_len == 0:
                     valid_len = 1

                 # Extract the unpadded sequence for all candidates in this batch item: shape (seq_len, 1, valid_len)
                 x_item = x[i:i+1, :, :valid_len].view(seq_len, 1, valid_len)

                 # Run the convolutions on the valid segment only for all candidates in this batch item
                 out_item = self.conv1(x_item)
                 out_item = self.relu1(out_item)
                 out_item = self.conv2(out_item)
                 out_item = self.relu2(out_item)

                 # Pool over the valid sequence
                 # The average is calculated securely without any padding tokens
                 pooled_item = out_item.mean(dim=2) # Shape: (seq_len, out_channels*2)
                 pooled_outs.append(pooled_item)

             pooled_out = torch.cat(pooled_outs, dim=0) # Shape: (batch_size * seq_len, out_channels*2)

        else:
             # Fast path if no mask is provided
             out = self.conv1(x_reshaped)
             out = self.relu1(out)
             out = self.conv2(out)
             out = self.relu2(out)
             pooled_out = out.mean(dim=2)

        embedded_x = self.linear(pooled_out)

        # Reshape back to the expected (batch_size, seq_len, embedding_dim)
        return embedded_x.view(batch_size, seq_len, self.embedding_dim)


class IdentityEmbedding(AbstractEmbedding):
    """Use Mallows rank features directly, without a learnable projection."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.embedding_dim:
            raise ValueError(
                "mallows_identity expects input [B,N,embedding_dim], "
                f"received {tuple(x.shape)} for embedding_dim={self.embedding_dim}."
            )
        return x


class MaskAwareBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d for ``[B,L,D]`` data with optional valid-token masking.

    Statistics are accumulated only over valid tokens.  The module keeps the
    usual ``weight``/``bias``/running-stat buffers, so evaluation with batch
    size one uses the standard BatchNorm running statistics.
    """

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError("MaskAwareBatchNorm1d expects [B,L,D] input.")
        if valid_mask is None:
            valid_mask = torch.ones(
                x.shape[:2], dtype=torch.bool, device=x.device
            )
        else:
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
            if valid_mask.shape != x.shape[:2]:
                raise ValueError(
                    "BN valid mask must match the [B,L] input dimensions."
                )

        if not self.training:
            return super().forward(x.transpose(1, 2).contiguous()).transpose(1, 2)

        mask = valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        count = mask.sum(dim=(0, 1))
        safe_count = count.clamp_min(1)
        mean = (x * mask).sum(dim=(0, 1)) / safe_count
        centered = (x - mean) * mask
        variance = (centered.square().sum(dim=(0, 1)) / safe_count).detach()

        if self.track_running_stats:
            with torch.no_grad():
                momentum = self.momentum if self.momentum is not None else 0.1
                self.running_mean.mul_(1.0 - momentum).add_(
                    mean.detach(), alpha=momentum
                )
                unbiased_variance = (
                    variance * count / (count - 1).clamp_min(1)
                )
                self.running_var.mul_(1.0 - momentum).add_(
                    unbiased_variance, alpha=momentum
                )
                self.num_batches_tracked.add_(1)

        scale = self.weight.to(dtype=x.dtype)
        shift = self.bias.to(dtype=x.dtype)
        output = (x - mean) * torch.rsqrt(variance + self.eps)
        output = output * scale + shift
        # Fully invalid token rows are padding and must stay finite.
        return output.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


def apply_batch_norm(
    x: torch.Tensor,
    batch_norm: nn.Module,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a norm to ``[B,L,D]`` with an optional mask-aware BN path."""
    if isinstance(batch_norm, MaskAwareBatchNorm1d):
        return batch_norm(x, valid_mask=valid_mask)
    return batch_norm(x)


def _packed_swiglu(projection: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU from one packed gate/value projection."""
    gate, value = projection(x).chunk(2, dim=-1)
    return F.silu(gate) * value




def _inverse_softplus(value: float) -> float:
    """Return a finite raw parameter whose softplus is ``value``."""
    if value <= 0:
        return -20.0
    return math.log(math.expm1(value))


class RotaryRankBias(nn.Module):
    """RARoPE plus ANLiBi parameters for one attention module."""

    def __init__(
        self,
        head_dim: int,
        numb_heads: int,
        use_rarope: bool,
        use_anlibi: bool,
        rarope_a: float,
        rarope_b: float,
        anlibi_alpha_init: float,
        anlibi_beta_init: float,
        anlibi_gamma: float,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RARoPE requires an even per-head dimension.")
        if rarope_a <= 0 or rarope_b <= 0:
            raise ValueError("RARoPE parameters a and b must be positive.")
        if anlibi_alpha_init < 0 or anlibi_beta_init < 0:
            raise ValueError("ANLiBi initial alpha and beta must be non-negative.")
        if anlibi_gamma <= 0:
            raise ValueError("ANLiBi gamma must be positive.")

        self.head_dim = head_dim
        self.use_rarope = use_rarope
        self.use_anlibi = use_anlibi
        self.rarope_a = float(rarope_a)
        self.rarope_b = float(rarope_b)
        self.anlibi_gamma = float(anlibi_gamma)
        frequencies = 10000.0 ** (
            -2.0 * torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        # Softplus enforces the required non-negative alpha/beta constraints.
        self.anlibi_alpha_raw = nn.Parameter(
            torch.full(
                (numb_heads,),
                _inverse_softplus(float(anlibi_alpha_init)),
            )
        )
        self.anlibi_beta_raw = nn.Parameter(
            torch.full(
                (numb_heads,),
                _inverse_softplus(float(anlibi_beta_init)),
            )
        )

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.anlibi_alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.anlibi_beta_raw)

    def nonlinear_position(
        self,
        positions: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate f(p), assigning the special position zero to no rotation."""
        positions = positions.to(device=reference.device, dtype=reference.dtype)
        return self.rarope_a * torch.log1p(
            self.rarope_b * positions.clamp_min(0.0)
        )

    def _rotate(
        self,
        tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_rarope:
            return tensor
        transformed = self.nonlinear_position(positions, tensor)
        angles = transformed.unsqueeze(-1) * self.frequencies
        cos = angles.cos()
        sin = angles.sin()
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return rotated.flatten(start_dim=-2)

    def transform_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._rotate(query, query_positions),
            self._rotate(key, key_positions),
        )

    def attention_bias(
        self,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_anlibi:
            return None
        query_float = query_positions.to(
            device=key_positions.device, dtype=torch.float32
        )
        key_float = key_positions.to(
            device=key_positions.device, dtype=torch.float32
        )
        key_formula = key_float.clamp_min(1.0).pow(-self.anlibi_gamma)
        rank_term = key_formula[:, None, None, :] * self.alpha[None, :, None, None]

        query_nonlinear = self.nonlinear_position(
            query_float, query_float
        )
        key_nonlinear = self.nonlinear_position(key_float, key_float)
        distance = (
            query_nonlinear[:, None, :, None]
            - key_nonlinear[:, None, None, :]
        ).abs()
        bias = rank_term - distance * self.beta[None, :, None, None]

        # START (position 0) is not an item rank.  Avoid both division by zero
        # and a synthetic rank preference by using neutral zero bias whenever
        # either side is special.
        ordinary_pair = (
            (query_float > 0).unsqueeze(2)
            & (key_float > 0).unsqueeze(1)
        )
        return bias.masked_fill(~ordinary_pair.unsqueeze(1), 0.0)


def _positioned_multihead_attention(
    mha: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
    position_module: Optional[RotaryRankBias] = None,
    query_positions: Optional[torch.Tensor] = None,
    key_positions: Optional[torch.Tensor] = None,
    clip_value: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an ``nn.MultiheadAttention`` with optional RARoPE/ANLiBi.

    Disabled position mechanisms take the original ``nn.MultiheadAttention``
    path.  Enabled mechanisms reuse its exact packed Q/K/V and output weights.
    """
    if position_module is None or (
        not position_module.use_rarope and not position_module.use_anlibi
    ):
        return mha(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

    tgt_len, batch_size, embed_dim = query.shape
    src_len = key.shape[0]
    numb_heads = mha.num_heads
    head_dim = embed_dim // numb_heads

    weight = mha.in_proj_weight
    bias = mha.in_proj_bias
    q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
    q_bias, k_bias, v_bias = bias.chunk(3, dim=0)
    q = F.linear(query, q_weight, q_bias)
    k = F.linear(key, k_weight, k_bias)
    v = F.linear(value, v_weight, v_bias)
    q = q.view(tgt_len, batch_size * numb_heads, head_dim).transpose(0, 1)
    k = k.view(src_len, batch_size * numb_heads, head_dim).transpose(0, 1)
    v = v.view(src_len, batch_size * numb_heads, head_dim).transpose(0, 1)

    if query_positions is None:
        query_positions = torch.zeros(
            batch_size, tgt_len, dtype=torch.long, device=query.device
        )
    if key_positions is None:
        key_positions = torch.zeros(
            batch_size, src_len, dtype=torch.long, device=key.device
        )
    query_positions_heads = query_positions[:, None, :].expand(
        batch_size, numb_heads, tgt_len
    ).reshape(batch_size * numb_heads, tgt_len)
    key_positions_heads = key_positions[:, None, :].expand(
        batch_size, numb_heads, src_len
    ).reshape(batch_size * numb_heads, src_len)
    q, k = position_module.transform_qk(
        q, k, query_positions_heads, key_positions_heads
    )

    scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(head_dim)
    if clip_value is not None:
        scores = clip_value * torch.tanh(scores)
    bias = position_module.attention_bias(query_positions, key_positions)
    if bias is not None:
        bias = bias.reshape(batch_size * numb_heads, tgt_len, src_len)
        scores = scores + bias

    if key_padding_mask is not None:
        invalid = key_padding_mask.to(device=scores.device, dtype=torch.bool)
        invalid = invalid[:, None, None, :].expand(-1, numb_heads, -1, src_len)
        invalid = invalid.reshape(batch_size * numb_heads, 1, src_len)
        scores = scores.masked_fill(invalid, float("-1e9"))

    weights = torch.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=mha.dropout, training=mha.training)
    output = torch.bmm(weights, v)
    output = output.transpose(0, 1).contiguous().view(
        tgt_len, batch_size, embed_dim
    )
    output = mha.out_proj(output)
    average_weights = weights.view(
        batch_size, numb_heads, tgt_len, src_len
    ).mean(dim=1)
    return output, average_weights


def calculate_consensus_positions(
    input_rankings: torch.Tensor,
    padding_mask: torch.Tensor,
    voter_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute deterministic one-based consensus item ranks from noisy input.

    The true center ranking is never consulted.  Padded rows receive position
    zero and are excluded from ranking.
    """
    batch_size, num_items, num_features = input_rankings.shape
    padding_mask = padding_mask.to(device=input_rankings.device, dtype=torch.bool)
    if voter_mask is None:
        voter_mask = torch.ones(
            batch_size,
            num_features,
            dtype=torch.bool,
            device=input_rankings.device,
        )
    else:
        voter_mask = voter_mask.to(device=input_rankings.device, dtype=torch.bool)
    voter_mask = voter_mask.reshape(batch_size, 1, num_features)

    feature_float = voter_mask.to(dtype=input_rankings.dtype)
    feature_count = feature_float.expand(
        batch_size, num_items, num_features
    ).sum(dim=-1).clamp_min(1.0)
    mean_rank = (
        input_rankings * feature_float
    ).sum(dim=-1) / feature_count
    mean_rank = mean_rank.masked_fill(padding_mask, float("inf"))
    order = torch.argsort(mean_rank, dim=1, stable=True)
    item_count = (~padding_mask).sum(dim=1)
    positions = torch.zeros(
        (batch_size, num_items),
        dtype=torch.long,
        device=input_rankings.device,
    )
    for batch in range(batch_size):
        count = int(item_count[batch])
        positions[batch, order[batch, :count]] = torch.arange(
            1, count + 1, device=input_rankings.device
        )
    return positions


class Transformer_encoder(nn.Module):
    """
    Encoder network based on self-attention transformer
    Inputs :
      h of size      (bsz, numb_items, dim_emb)    batch of input candidates
      src_key_padding_mask of size (bsz, numb_items) batch of masks for padding
    Outputs :
      h of size      (bsz, numb_items, dim_emb)    batch of encoded candidates
      weights of size  (bsz, numb_heads, numb_items, numb_items) batch of attention scores
      src_key_padding_mask of size (bsz, numb_items) batch of masks for padding (returned for convenience)
    """

    def __init__(
        self,
        numb_layers,
        dim_emb,
        numb_heads,
        dim_ff,
        batchnorm: bool,
        norm_position: str = "pre",
        use_rarope: bool = True,
        use_anlibi: bool = True,
        rarope_a: float = 1.0,
        rarope_b: float = 1.0,
        anlibi_alpha_init: float = 0.01,
        anlibi_beta_init: float = 0.01,
        anlibi_gamma: float = 1.0,
    ):
        super(Transformer_encoder, self).__init__()
        assert dim_emb % numb_heads == 0, "dim_emb must be divisible by numb_heads"
        self.MHA_layers = nn.ModuleList([nn.MultiheadAttention(dim_emb, numb_heads) for _ in range(numb_layers)])
        self.linear1_layers = nn.ModuleList([nn.Linear(dim_emb, 2 * dim_ff) for _ in range(numb_layers)])
        self.linear2_layers = nn.ModuleList([nn.Linear(dim_ff, dim_emb) for _ in range(numb_layers)])
        if batchnorm:
            self.norm1_layers = nn.ModuleList([MaskAwareBatchNorm1d(dim_emb) for _ in range(numb_layers)])
            self.norm2_layers = nn.ModuleList([MaskAwareBatchNorm1d(dim_emb) for _ in range(numb_layers)])
        else:
            self.norm1_layers = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(numb_layers)])
            self.norm2_layers = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(numb_layers)])
        self.position_modules = nn.ModuleList([
            RotaryRankBias(
                head_dim=dim_emb // numb_heads,
                numb_heads=numb_heads,
                use_rarope=use_rarope,
                use_anlibi=use_anlibi,
                rarope_a=rarope_a,
                rarope_b=rarope_b,
                anlibi_alpha_init=anlibi_alpha_init,
                anlibi_beta_init=anlibi_beta_init,
                anlibi_gamma=anlibi_gamma,
            )
            for _ in range(numb_layers)
        ])

        self.numb_layers = numb_layers
        self.numb_heads = numb_heads
        self.batchnorm = batchnorm
        if norm_position not in ("pre", "post"):
            raise ValueError("norm_position must be 'pre' or 'post'.")
        self.norm_position = norm_position

    def forward(
        self,
        h: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the encoder.

        Args:
            h (torch.Tensor): Input tensor. Shape: (bsz, seq_len, dim_emb).
            src_key_padding_mask (torch.Tensor): Padding mask. Shape: (bsz, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The encoded tensor. Shape: (batch_size, seq_len, dim_emb).
                - The attention weights. Shape: (batch_size, seq_len, seq_len).
                - The input padding mask. Shape: (batch_size, seq_len).
        """
        if positions is None:
            positions = torch.zeros(
                h.shape[:2], dtype=torch.long, device=h.device
            )
        weights = None # To store weights from the last layer
        h_batch_first = h

        for i in range(self.numb_layers):
            if self.norm_position == "pre":
                attention_input = apply_batch_norm(
                    h_batch_first,
                    self.norm1_layers[i],
                    valid_mask=~src_key_padding_mask,
                ).transpose(0, 1)
                h_att, weights = _positioned_multihead_attention(
                    self.MHA_layers[i],
                    attention_input,
                    attention_input,
                    attention_input,
                    src_key_padding_mask,
                    self.position_modules[i],
                    positions,
                    positions,
                )
                h_batch_first = h_batch_first + h_att.transpose(0, 1)

                ffn_input = apply_batch_norm(
                    h_batch_first,
                    self.norm2_layers[i],
                    valid_mask=~src_key_padding_mask,
                )
                h_batch_first = h_batch_first + self.linear2_layers[i](
                    _packed_swiglu(self.linear1_layers[i], ffn_input)
                )
            else:
                h = h_batch_first.transpose(0, 1)
                h_att, weights = _positioned_multihead_attention(
                    self.MHA_layers[i],
                    h,
                    h,
                    h,
                    src_key_padding_mask,
                    self.position_modules[i],
                    positions,
                    positions,
                )
                h_batch_first = h_batch_first + h_att.transpose(0, 1)
                h_batch_first = apply_batch_norm(
                    h_batch_first,
                    self.norm1_layers[i],
                    valid_mask=~src_key_padding_mask,
                )
                h = h_batch_first.transpose(0, 1)
                h_ff = self.linear2_layers[i](
                    _packed_swiglu(self.linear1_layers[i], h)
                )
                h_batch_first = h_batch_first + h_ff.transpose(0, 1)
                h_batch_first = apply_batch_norm(
                    h_batch_first,
                    self.norm2_layers[i],
                    valid_mask=~src_key_padding_mask,
                )

        return h_batch_first, weights, src_key_padding_mask


def newMHA(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
           numb_heads: int, mask: Optional[torch.Tensor] = None,
           clip_value: Optional[float] = None,
           position_module: Optional[RotaryRankBias] = None,
           query_positions: Optional[torch.Tensor] = None,
           key_positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute multi-head attention (MHA) given a query Q, key K, value V and attention mask :
      h = Concat_{k=1}^numb_heads softmax(Q_k^T.K_k).V_k
    Note : We did not use nn.MultiheadAttention to avoid re-computing all linear transformations at each call.

    Inputs :
      Q of size (bsz, seq_len_q, dim_emb)           batch of queries
      K of size (bsz, seq_len_kv, dim_emb)          batch of keys
      V of size (bsz, seq_len_kv, dim_emb)          batch of values
      mask of size (bsz, seq_len_kv)                batch of masks of visited cities
      clip_value is a scalar
    Outputs :
      attn_output of size (bsz, seq_len_q, dim_emb) batch of attention vectors
      attn_weights of size (bsz, seq_len_q, seq_len_kv) batch of attention weights
    """

    bsz, numb_items, emd_dim = K.size()
    seq_len_q = Q.size(1)

    # Reshape Q, K, V for multi-head computation
    if numb_heads > 1:
        Q = Q.transpose(1, 2).contiguous().view(bsz * numb_heads, emd_dim // numb_heads, seq_len_q).transpose(1, 2)
        K = K.transpose(1, 2).contiguous().view(bsz * numb_heads, emd_dim // numb_heads, numb_items).transpose(1, 2)
        V = V.transpose(1, 2).contiguous().view(bsz * numb_heads, emd_dim // numb_heads, numb_items).transpose(1, 2)

    if position_module is not None:
        if query_positions is None:
            query_positions = torch.zeros(
                bsz, seq_len_q, dtype=torch.long, device=Q.device
            )
        if key_positions is None:
            key_positions = torch.zeros(
                bsz, numb_items, dtype=torch.long, device=K.device
            )
        query_positions_heads = query_positions[:, None, :].expand(
            bsz, numb_heads, seq_len_q
        ).reshape(bsz * numb_heads, seq_len_q)
        key_positions_heads = key_positions[:, None, :].expand(
            bsz, numb_heads, numb_items
        ).reshape(bsz * numb_heads, numb_items)
        Q, K = position_module.transform_qk(
            Q, K, query_positions_heads, key_positions_heads
        )
    query_positions_for_bias = query_positions
    key_positions_for_bias = key_positions

    # Calculate attention scores
    attn_weights = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
    if position_module is not None and position_module.use_anlibi:
        bias = position_module.attention_bias(
            query_positions_for_bias, key_positions_for_bias
        )
        if bias is not None:
            if numb_heads > 1:
                bias = bias.reshape(bsz * numb_heads, seq_len_q, numb_items)
            attn_weights = attn_weights + bias

    if clip_value is not None:
        attn_weights = clip_value * torch.tanh(attn_weights)

    if mask is not None:
        if numb_heads > 1:
            mask = torch.repeat_interleave(mask, repeats=numb_heads, dim=0)
        # Apply mask by setting masked positions to negative infinity
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-1e9'))

    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.bmm(attn_weights, V)

    if numb_heads > 1:
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, emd_dim, seq_len_q).transpose(1, 2)
        attn_weights = attn_weights.view(bsz, numb_heads, seq_len_q, numb_items).mean(dim=1)

    return attn_output, attn_weights


class AutoRegressiveDecoderLayer(nn.Module):
    """
    Single decoder layer based on self-attention and query-attention
    Inputs :
      h_t of size     (bsz, 1, dim_emb)             batch of input queries
      K_att of size   (bsz, numb_items, dim_emb) batch of query-attention keys
      V_att of size   (bsz, numb_items, dim_emb) batch of query-attention values
      mask of size    (bsz, numb_items)        batch of masks of visited cities
    Output :
      h_t of size (bsz, dim_emb)                    batch of transformed queries
    """

    def __init__(
        self,
        dim_emb,
        numb_heads,
        batchnorm: bool = True,
        norm_position: str = "pre",
        use_rarope: bool = True,
        use_anlibi: bool = True,
        rarope_a: float = 1.0,
        rarope_b: float = 1.0,
        anlibi_alpha_init: float = 0.01,
        anlibi_beta_init: float = 0.01,
        anlibi_gamma: float = 1.0,
    ):
        super(AutoRegressiveDecoderLayer, self).__init__()
        self.dim_emb = dim_emb
        self.numb_heads = numb_heads
        self.norm_position = norm_position
        # Self-attention layers
        self.Wq_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wk_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wv_selfatt = nn.Linear(dim_emb, dim_emb)
        self.W0_selfatt = nn.Linear(dim_emb, dim_emb)
        # Cross-attention layers
        self.W0_att = nn.Linear(dim_emb, dim_emb)
        self.Wq_att = nn.Linear(dim_emb, dim_emb)
        # Feed-forward layers
        self.W1_MLP = nn.Linear(dim_emb, 2 * dim_emb)
        self.W2_MLP = nn.Linear(dim_emb, dim_emb)
        # Preserve parameter names while switching the target path to Pre-BN.
        if batchnorm:
            self.BN_selfatt = MaskAwareBatchNorm1d(dim_emb)
            self.BN_att = MaskAwareBatchNorm1d(dim_emb)
            self.BN_MLP = MaskAwareBatchNorm1d(dim_emb)
        else:
            self.BN_selfatt = nn.LayerNorm(dim_emb)
            self.BN_att = nn.LayerNorm(dim_emb)
            self.BN_MLP = nn.LayerNorm(dim_emb)
        self.position_selfatt = RotaryRankBias(
            head_dim=dim_emb // numb_heads,
            numb_heads=numb_heads,
            use_rarope=use_rarope,
            use_anlibi=use_anlibi,
            rarope_a=rarope_a,
            rarope_b=rarope_b,
            anlibi_alpha_init=anlibi_alpha_init,
            anlibi_beta_init=anlibi_beta_init,
            anlibi_gamma=anlibi_gamma,
        )
        self.position_crossatt = RotaryRankBias(
            head_dim=dim_emb // numb_heads,
            numb_heads=numb_heads,
            use_rarope=use_rarope,
            use_anlibi=use_anlibi,
            rarope_a=rarope_a,
            rarope_b=rarope_b,
            anlibi_alpha_init=anlibi_alpha_init,
            anlibi_beta_init=anlibi_beta_init,
            anlibi_gamma=anlibi_gamma,
        )
        # Cached keys and values for self-attention
        self.K_selfatt = None
        self.V_selfatt = None
        self.K_selfatt_positions = None

    def reset_selfatt_keys_values(self):
        """Resets the cached self-attention keys and values."""
        self.K_selfatt = None
        self.V_selfatt = None
        self.K_selfatt_positions = None

    def forward(
        self,
        h_t: torch.Tensor,
        K_att: torch.Tensor,
        V_att: torch.Tensor,
        mask: torch.Tensor,
        query_position: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the autoregressive decoder layer."""
        bsz = h_t.size(0)
        # Ensure h_t is (bsz, 1, dim_emb) for sequence processing
        h_t = h_t.view(bsz, 1, self.dim_emb)
        if query_position is None:
            query_position = torch.zeros(
                bsz, dtype=torch.long, device=h_t.device
            )
        query_position = query_position.reshape(-1)

        # 1. Masked Self-Attention over the decoded sequence
        self_input = (
            apply_batch_norm(h_t, self.BN_selfatt)
            if self.norm_position == "pre"
            else h_t
        )
        q_selfatt = self.Wq_selfatt(self_input)
        k_selfatt = self.Wk_selfatt(self_input)
        v_selfatt = self.Wv_selfatt(self_input)

        # Concatenate the new self-attention key and value to the previous keys and values
        if self.K_selfatt is None:
            self.K_selfatt = k_selfatt
            self.V_selfatt = v_selfatt
            self.K_selfatt_positions = query_position[:, None]
        else:
            self.K_selfatt = torch.cat([self.K_selfatt, k_selfatt], dim=1)
            self.V_selfatt = torch.cat([self.V_selfatt, v_selfatt], dim=1)
            self.K_selfatt_positions = torch.cat(
                [self.K_selfatt_positions, query_position[:, None]], dim=1
            )

        # Compute self-attention between candidates in the partial ranking
        h_t_res = h_t
        # No mask needed for self-attention as we attend to all previously decoded items
        h_t_att, _ = newMHA(
            q_selfatt,
            self.K_selfatt,
            self.V_selfatt,
            self.numb_heads,
            position_module=self.position_selfatt,
            query_positions=query_position[:, None],
            key_positions=self.K_selfatt_positions,
        )
        h_t = h_t_res + self.W0_selfatt(h_t_att) #size(h_t)=(bsz,1,dim_emb)
        if self.norm_position == "post":
            h_t = self.BN_selfatt(h_t.squeeze(1)).view(bsz, 1, self.dim_emb)

        # 2. Cross-Attention over encoder outputs
        h_t_res = h_t
        cross_input = (
            apply_batch_norm(h_t, self.BN_att)
            if self.norm_position == "pre"
            else h_t
        )
        q_a = self.Wq_att(cross_input)  # size(q_a)=(bsz, 1, bem_dim)
        # Use the provided mask to ignore padded/ranked candidates from the encoder
        h_t_att, _ = newMHA(
            q_a,
            K_att,
            V_att,
            self.numb_heads,
            mask,
            position_module=self.position_crossatt,
            query_positions=query_position[:, None],
            key_positions=key_positions,
        )
        h_t = h_t_res + self.W0_att(h_t_att)  # size(h_t)=(bsz, 1, dim_emb)
        if self.norm_position == "post":
            h_t = self.BN_att(h_t.squeeze(1)).view(bsz, 1, self.dim_emb)

        # 3. MLP (Feed-forward)
        h_t_res = h_t
        ffn_input = (
            apply_batch_norm(h_t, self.BN_MLP)
            if self.norm_position == "pre"
            else h_t
        )
        h_t_mlp = self.W2_MLP(_packed_swiglu(self.W1_MLP, ffn_input))
        h_t = h_t_res + h_t_mlp
        if self.norm_position == "post":
            h_t = self.BN_MLP(h_t.squeeze(1))
        else:
            h_t = h_t.squeeze(1)
        return h_t


class Transformer_decoder(nn.Module):
    """
    Decoder network based on self-attention and query-attention transformers
    Inputs :
        h_t of size     (bsz, dim_emb)                         batch of input queries
        K_att of size   (bsz, numb_items, dim_emb*numb_layers_decoder) batch of query-attention keys for all decoding layers
        V_att of size   (bsz, numb_items, dim_emb*numb_layers_decoder) batch of query-attention values for all decoding layers
        mask of size    (bsz, numb_items)                 batch of masks of already ranked candidates
    Output :
        prob_next_candidates of size (bsz, numb_items)    batch of probabilities of next node
    """

    def __init__(
        self,
        dim_emb,
        numb_heads,
        numb_layers_decoder,
        batchnorm: bool = True,
        norm_position: str = "pre",
        use_rarope: bool = True,
        use_anlibi: bool = True,
        rarope_a: float = 1.0,
        rarope_b: float = 1.0,
        anlibi_alpha_init: float = 0.01,
        anlibi_beta_init: float = 0.01,
        anlibi_gamma: float = 1.0,
    ):
        super(Transformer_decoder, self).__init__()
        self.dim_emb = dim_emb
        self.numb_heads = numb_heads
        self.numb_layers_decoder = numb_layers_decoder
        if numb_layers_decoder < 1:
            raise ValueError("numb_layers_decoder must be at least 1.")

        # N-1 standard decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                AutoRegressiveDecoderLayer(
                    dim_emb,
                    numb_heads,
                    batchnorm=batchnorm,
                    norm_position=norm_position,
                    use_rarope=use_rarope,
                    use_anlibi=use_anlibi,
                    rarope_a=rarope_a,
                    rarope_b=rarope_b,
                    anlibi_alpha_init=anlibi_alpha_init,
                    anlibi_beta_init=anlibi_beta_init,
                    anlibi_gamma=anlibi_gamma,
                )
                for _ in range(numb_layers_decoder - 1)
            ]
        )
        # Final layer to project query for attention-based probability calculation
        self.Wq_final = nn.Linear(dim_emb, dim_emb)

    # Reset to None self-attention keys and values when decoding starts
    def reset_selfatt_keys_values(self):
        """Resets the cached self-attention keys and values for all layers."""
        for l in range(self.numb_layers_decoder - 1):
            self.decoder_layers[l].reset_selfatt_keys_values()

    def forward(
        self,
        h_t: torch.Tensor,
        K_att: torch.Tensor,
        V_att: torch.Tensor,
        mask: torch.Tensor,
        query_position: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the Transformer decoder stack."""
        # Pass through N-1 standard decoder layers
        for l in range(self.numb_layers_decoder - 1):
            # Get the K, V slices for this layer
            K_att_l = K_att[:, :, l * self.dim_emb:(l + 1) * self.dim_emb].contiguous()
            V_att_l = V_att[:, :, l * self.dim_emb:(l + 1) * self.dim_emb].contiguous()
            h_t = self.decoder_layers[l](
                h_t,
                K_att_l,
                V_att_l,
                mask,
                query_position=query_position,
                key_positions=key_positions,
            )

        # Final (Nth) layer: single-head attention for probability output
        l = self.numb_layers_decoder - 1 # Get index for the final K/V slice
        K_att_l = K_att[:, :, l * self.dim_emb:(l + 1) * self.dim_emb].contiguous()
        V_att_l = V_att[:, :, l * self.dim_emb:(l + 1) * self.dim_emb].contiguous()

        q_final = self.Wq_final(h_t)
        bsz = h_t.size(0)
        q_final = q_final.view(bsz, 1, self.dim_emb)

        # Get attention weights from the final layer. Use 1 head.
        _, att_weights = newMHA(q_final, K_att_l, V_att_l, 1, mask, clip_value=10.0)
        prob_next_node = att_weights.squeeze(1) # Shape: (bsz, numb_items)
        return prob_next_node


def generate_positional_encodeing(d_model, max_len):
    """
    Create standard transformer PEs.
    Inputs :
      d_model is a scalar correspoding to the hidden dimension
      max_len is the maximum length of the sequence
    Output :
      pe of size (max_len, d_model), where d_model=dim_emb, max_len=1000
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class kemeny_transformer(nn.Module):
    """Model
    The kemeny_transformer network is composed of two steps :
      Step 1. Encoder step : Take a set of (voters,numb_items) base rankings
                             and encode the set with self-transformer.
      Step 2. Decoder step : Build the kemeny-optimal ranking recursively/autoregressively,
                             i.e. one kandidate at a time, with a self-transformer and query-transformer.
    Inputs :
      x of size (bsz, numb_items, dim_input_candidates) ranking position of candidates on base rankings
      deterministic is a boolean : If True the kemeny-optimal ranking will choose the city with the highest probability.
                                   If False the salesman will choose the city with Bernouilli sampling.
    Outputs :
      ranking of size (bsz, candidates) : batch of final ranking, i.e. sequences of ordered candidate
                                          ranking [b,t] contains the idx of the candidate picked at step t in batch b
      sumLogProbOfActions of size (bsz,) : batch of sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
    """

    def __init__(self,
                 embedding_type: EmbeddingType,
                 input_dim: int,
                 embedding_dim: int,
                 dim_ff: int,
                 numb_heads: int,
                 numb_layers_decoder: int,
                 numb_layers_encoder:int,
                 max_len_PE: int,
                 conv_out_channels: int = 64,
                 batchnorm: bool = True,
                 norm_type: Optional[str] = None,
                 norm_position: str = "pre",
                 num_generated_rankings: Optional[int] = None,
                 mallows_theta: float = 1.0,
                 mallows_seed: Optional[int] = None,
                 mallows_resample_each_epoch: bool = True,
                 use_rarope: bool = True,
                 use_anlibi: bool = True,
                 rarope_a: float = 1.0,
                 rarope_b: float = 1.0,
                 anlibi_alpha_init: float = 0.01,
                 anlibi_beta_init: float = 0.01,
                 anlibi_gamma: float = 1.0,
                 use_original_decoder_pe: bool = False,
                 use_rarope_encoder: Optional[bool] = None,
                 use_rarope_decoder_self: Optional[bool] = None,
                 use_rarope_decoder_cross: Optional[bool] = None,
                 use_anlibi_encoder: Optional[bool] = None,
                 use_anlibi_decoder_self: Optional[bool] = None,
                 use_anlibi_decoder_cross: Optional[bool] = None,
                 training_mode: str = "teacher_forcing",
                 loss_type: str = "cross_entropy",
                 preserve_original_mask_update: bool = True):
        super(kemeny_transformer, self).__init__()
        if isinstance(embedding_type, str):
            embedding_type = EmbeddingType(embedding_type.lower())
        if not isinstance(embedding_type, EmbeddingType):
            try:
                embedding_type = EmbeddingType(embedding_type.value)
            except AttributeError as error:
                raise ValueError(f"Unknown embedding_type: {embedding_type!r}") from error
        self.embedding_type=embedding_type
        self.dim_emb = embedding_dim
        self.input_dim=input_dim
        self.dim_ff=dim_ff
        if norm_type is None:
            norm_type = "batchnorm" if batchnorm else "layernorm"
        if norm_type not in ("batchnorm", "layernorm"):
            raise ValueError("norm_type must be 'batchnorm' or 'layernorm'.")
        batchnorm = norm_type == "batchnorm"
        if norm_position not in ("pre", "post"):
            raise ValueError("norm_position must be 'pre' or 'post'.")
        if training_mode != "teacher_forcing":
            raise ValueError("New supervised mode requires training_mode='teacher_forcing'.")
        if loss_type != "cross_entropy":
            raise ValueError("New supervised mode requires loss_type='cross_entropy'.")
        if not preserve_original_mask_update:
            raise ValueError("The original Mask Vector Update must be preserved.")
        self.norm_type = norm_type
        self.norm_position = norm_position
        self.training_mode = training_mode
        self.loss_type = loss_type
        self.preserve_original_mask_update = preserve_original_mask_update

        # --- 1. Embedding Layer (Instantiated based on type) ---
        if embedding_type == EmbeddingType.LINEAR:
            # For LinearEmbedding, input_dim is required and fixed.
            self.embedding_layer = LinearEmbedding(input_dim=input_dim, embedding_dim=embedding_dim)
            print(f"linear embedding model with input dimension {input_dim}, embedding dimension {embedding_dim}")
        elif embedding_type == EmbeddingType.CONV:


            # For ConvEmbedding, input_dim is not needed for initialization,
            # as it can handle variable input lengths.
            self.embedding_layer = ConvEmbedding(embedding_dim=embedding_dim, out_channels=conv_out_channels)
            print(f"conv embedding model with embedding dimension {embedding_dim}, out channels {conv_out_channels}")
        elif embedding_type == EmbeddingType.MALLOWS_IDENTITY:
            if (
                num_generated_rankings is not None
                and int(num_generated_rankings) != embedding_dim
            ):
                raise ValueError(
                    "num_generated_rankings must equal embedding_dim in "
                    "mallows_identity mode."
                )
            num_generated_rankings = embedding_dim
            self.embedding_layer = IdentityEmbedding(embedding_dim=embedding_dim)
            print(
                "mallows identity model with "
                f"{num_generated_rankings} generated rankings"
            )
        else:
            raise ValueError(f"Unknown embedding_type: '{embedding_type}'. Please use the EmbeddingType enum.")
        print(embedding_type)
        self.num_generated_rankings = (
            int(num_generated_rankings)
            if num_generated_rankings is not None
            else embedding_dim
        )
        if self.num_generated_rankings <= 0:
            raise ValueError("num_generated_rankings must be positive.")
        self.mallows_theta = float(mallows_theta)
        self.mallows_resample_each_epoch = bool(mallows_resample_each_epoch)
        self.mallows_sampler = MallowsPhiSampler(
            theta=self.mallows_theta, seed=mallows_seed
        )

        use_rarope_encoder = (
            use_rarope if use_rarope_encoder is None else use_rarope_encoder
        )
        use_rarope_decoder_self = (
            use_rarope
            if use_rarope_decoder_self is None
            else use_rarope_decoder_self
        )
        use_rarope_decoder_cross = (
            use_rarope
            if use_rarope_decoder_cross is None
            else use_rarope_decoder_cross
        )
        use_anlibi_encoder = (
            use_anlibi if use_anlibi_encoder is None else use_anlibi_encoder
        )
        use_anlibi_decoder_self = (
            use_anlibi
            if use_anlibi_decoder_self is None
            else use_anlibi_decoder_self
        )
        use_anlibi_decoder_cross = (
            use_anlibi
            if use_anlibi_decoder_cross is None
            else use_anlibi_decoder_cross
        )
        self.use_original_decoder_pe = bool(use_original_decoder_pe)
        # --- 2. Encoder ---
        self.encoder = Transformer_encoder(
            numb_layers_encoder,
            embedding_dim,
            numb_heads,
            dim_ff,
            batchnorm,
            norm_position=norm_position,
            use_rarope=use_rarope_encoder,
            use_anlibi=use_anlibi_encoder,
            rarope_a=rarope_a,
            rarope_b=rarope_b,
            anlibi_alpha_init=anlibi_alpha_init,
            anlibi_beta_init=anlibi_beta_init,
            anlibi_gamma=anlibi_gamma,
        )

        # --- 3. Decoder Placeholder ---
        # vector to start decoding
        self.start_placeholder = nn.Parameter(torch.randn(embedding_dim))


        # --- 4. Decoder ---
        self.decoder = Transformer_decoder(
            embedding_dim,
            numb_heads,
            numb_layers_decoder,
            batchnorm=batchnorm,
            norm_position=norm_position,
            use_rarope=use_rarope_decoder_self,
            use_anlibi=use_anlibi_decoder_self,
            rarope_a=rarope_a,
            rarope_b=rarope_b,
            anlibi_alpha_init=anlibi_alpha_init,
            anlibi_beta_init=anlibi_beta_init,
            anlibi_gamma=anlibi_gamma,
        )
        # Cross-attention has an independent ablation switch.
        for layer in self.decoder.decoder_layers:
            layer.position_crossatt.use_rarope = bool(use_rarope_decoder_cross)
            layer.position_crossatt.use_anlibi = bool(use_anlibi_decoder_cross)

        # --- 5. Decoder K/V Projections ---
        # Linear layers to project encoder outputs into keys and values for decoder's cross-attention.
        self.WK_att_decoder = nn.Linear(embedding_dim, numb_layers_decoder * embedding_dim)
        self.WV_att_decoder = nn.Linear(embedding_dim, numb_layers_decoder * embedding_dim)

        # --- 6. Positional Encoding ---
        self.register_buffer(
            "PE",
            generate_positional_encodeing(embedding_dim, max_len_PE),
            persistent=False,
        )

    def generate_mallows_inputs(
        self,
        true_rankings: torch.Tensor,
        num_rankings: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create the model input ``[B,N,d]`` from one center ranking per row.

        For variable-length items, supply the usual ``[B,N]`` padding mask.
        Valid item IDs must be contiguous ``0..N_valid-1`` within each row;
        padded item rows are filled with zero rank features.
        """
        generated = (
            self.num_generated_rankings
            if num_rankings is None
            else int(num_rankings)
        )
        if generated != self.dim_emb:
            raise ValueError("The number of generated rankings must equal dim_emb.")
        if padding_mask is not None:
            if true_rankings.shape != padding_mask.shape:
                raise ValueError("true_rankings and padding_mask must both be [B,N].")
            padding_mask = padding_mask.to(
                device=true_rankings.device, dtype=torch.bool
            )
            batch_size, num_items = true_rankings.shape
            output = true_rankings.new_zeros(
                (batch_size, num_items, generated), dtype=torch.float32
            )
            for batch in range(batch_size):
                valid_count = int((~padding_mask[batch]).sum().item())
                if valid_count == 0:
                    continue
                valid_center = true_rankings[batch, :valid_count]
                valid_noisy = self.mallows_sampler.sample_position_matrix(
                    valid_center.unsqueeze(0), generated
                )[0]
                output[batch, :valid_count] = valid_noisy.to(output.device)
            return output
        noisy_rankings = self.mallows_sampler.sample_position_matrix(
            true_rankings, generated
        )
        return noisy_rankings.to(device=true_rankings.device)

    def _embed(self, x: torch.Tensor, voter_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if isinstance(self.embedding_layer, ConvEmbedding):
            return self.embedding_layer(x, voter_mask=voter_mask)
        return self.embedding_layer(x)

    def _encode(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        voter_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode noisy rankings and prepare decoder K/V and rank positions.

        The true target ranking is not accepted by this method.
        """
        if x.dim() != 3:
            raise ValueError("x must have shape [batch, num_items, features].")
        if padding_mask.shape != x.shape[:2]:
            raise ValueError("padding_mask must have shape [batch, num_items].")
        padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)
        if self.embedding_type == EmbeddingType.MALLOWS_IDENTITY:
            if x.size(-1) != self.dim_emb:
                raise ValueError(
                    f"Expected last input dimension {self.dim_emb}, got {x.size(-1)}."
                )
        bsz, numb_candidates = x.shape[:2]
        device = x.device

        # Consensus positions are derived only from observed noisy rankings.
        consensus_positions = calculate_consensus_positions(
            x, padding_mask, voter_mask
        )
        h = self._embed(x, voter_mask)
        h = torch.cat(
            [h, self.start_placeholder.repeat(bsz, 1, 1)], dim=1
        )
        encoder_padding_mask = torch.cat(
            [
                padding_mask,
                torch.zeros(bsz, 1, device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        encoder_positions = torch.cat(
            [
                consensus_positions[:, :numb_candidates],
                torch.zeros(bsz, 1, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        h_encoder, _, _ = self.encoder(
            h,
            src_key_padding_mask=encoder_padding_mask,
            positions=encoder_positions,
        )
        K_att_decoder = self.WK_att_decoder(h_encoder)
        V_att_decoder = self.WV_att_decoder(h_encoder)
        return h_encoder, K_att_decoder, V_att_decoder, encoder_positions

    def _prepare_decoder_state(
        self,
        h_encoder: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, encoder_length, _ = h_encoder.shape
        device = h_encoder.device
        rows = torch.arange(bsz, device=device)
        start_index = encoder_length - 1
        h_t = h_encoder[:, start_index, :]
        if self.use_original_decoder_pe:
            h_t = h_t + self.PE[0].unsqueeze(0)
        mask_ranked_candidates = torch.zeros(
            bsz, encoder_length, device=device, dtype=torch.bool
        )
        mask_ranked_candidates[rows, start_index] = True
        self.decoder.reset_selfatt_keys_values()
        return h_t, mask_ranked_candidates, encoder_padding_mask

    def _next_decoder_input(
        self,
        h_encoder: torch.Tensor,
        item_indices: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        rows = torch.arange(h_encoder.size(0), device=h_encoder.device)
        h_t = h_encoder[rows, item_indices, :]
        if self.use_original_decoder_pe:
            if step + 1 >= self.PE.size(0):
                raise ValueError("max_len_PE is too small for this ranking.")
            h_t = h_t + self.PE[step + 1].unsqueeze(0)
        return h_t

    def training_forward_from_dataset(
        self,
        dataset: Union[torch.Tensor, Mapping[str, torch.Tensor]],
        true_rankings: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        voter_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run supervised Teacher Forcing on a model-ready dataset batch.

        ``noisy_positions`` is the Encoder input and has shape ``[B,S,d]``:
        ``B`` is batch size, ``S`` is the number of items, and ``d`` is the
        number of noisy rankings per true ranking. ``true_rankings`` is the
        Decoder Teacher Forcing target and has shape ``[B,S]`` with zero-based
        item IDs.

        Args:
            dataset: Either a mapping containing ``noisy_positions`` and
                ``true_rankings``, or the Encoder input tensor itself.
            true_rankings: Decoder targets when ``dataset`` is a tensor.
            padding_mask: Optional ``[B,S]`` boolean mask; ``True`` marks
                padding. If omitted, every item is treated as valid.
            voter_mask: Optional mask forwarded to the Encoder.

        Returns:
            The same loss and ranking dictionary produced by
            :meth:`teacher_forcing_forward`.
        """
        if isinstance(dataset, Mapping):
            noisy_positions = dataset.get("noisy_positions")
            if noisy_positions is None:
                raise ValueError(
                    "Dataset batch must contain 'noisy_positions' with shape "
                    "[B,S,d]. The ordering tensor 'noisy_rankings' has shape "
                    "[B,d,S] and is not the Encoder input."
                )
            if true_rankings is not None:
                raise ValueError(
                    "Pass true_rankings either in the dataset mapping or as "
                    "the second argument, not both."
                )
            true_rankings = dataset.get("true_rankings")
            if true_rankings is None:
                raise ValueError(
                    "Dataset batch must contain 'true_rankings'."
                )
            if padding_mask is None:
                padding_mask = dataset.get("padding_mask")
        else:
            noisy_positions = dataset

        if not isinstance(noisy_positions, torch.Tensor):
            raise TypeError("noisy_positions must be a torch.Tensor.")
        if not isinstance(true_rankings, torch.Tensor):
            raise TypeError(
                "true_rankings must be provided as a torch.Tensor."
            )
        if noisy_positions.dim() != 3:
            raise ValueError(
                "noisy_positions must have shape [B,S,d]."
            )
        if true_rankings.dim() != 2:
            raise ValueError(
                "true_rankings must have shape [B,S]."
            )

        batch_size, num_items, num_noisy_rankings = noisy_positions.shape
        if true_rankings.shape != (batch_size, num_items):
            raise ValueError(
                "noisy_positions and true_rankings must agree on [B,S]: "
                f"got {tuple(noisy_positions.shape)} and "
                f"{tuple(true_rankings.shape)}."
            )
        if num_noisy_rankings != self.dim_emb:
            raise ValueError(
                "The Encoder input dimension must equal embedding_dim: "
                f"got d={num_noisy_rankings}, embedding_dim={self.dim_emb}. "
                "Set NUM_NOISY_RANKINGS to embedding_dim when generating data."
            )

        device = self.start_placeholder.device
        dtype = self.start_placeholder.dtype
        noisy_positions = noisy_positions.to(device=device, dtype=dtype)
        true_rankings = true_rankings.to(device=device, dtype=torch.long)

        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch_size, num_items),
                device=device,
                dtype=torch.bool,
            )
        else:
            if not isinstance(padding_mask, torch.Tensor):
                raise TypeError("padding_mask must be a torch.Tensor.")
            if padding_mask.shape != (batch_size, num_items):
                raise ValueError("padding_mask must have shape [B,S].")
            padding_mask = padding_mask.to(device=device, dtype=torch.bool)

        valid_observed = noisy_positions[~padding_mask]
        if valid_observed.numel():
            if not bool(torch.isfinite(valid_observed).all()):
                raise ValueError(
                    "noisy_positions contains NaN or infinite values."
                )
            if bool(
                (valid_observed < 1).any()
                or (valid_observed > num_items).any()
            ):
                raise ValueError(
                    "Valid noisy_positions values must be one-based ranks in "
                    "[1, num_items]."
                )

        for batch_index in range(batch_size):
            valid_target = true_rankings[
                batch_index, ~padding_mask[batch_index]
            ]
            if valid_target.numel() == 0:
                continue
            if bool(
                (valid_target < 0).any()
                or (valid_target >= num_items).any()
            ):
                raise ValueError(
                    "Valid true_rankings values must be zero-based item IDs "
                    "in [0, num_items-1]."
                )
            if torch.unique(valid_target).numel() != valid_target.numel():
                raise ValueError(
                    "Each valid true ranking must contain unique item IDs."
                )

        return self.teacher_forcing_forward(
            noisy_positions,
            true_rankings,
            padding_mask,
            voter_mask=voter_mask,
        )

    def forward(self,  x: torch.Tensor, padding_mask: torch.Tensor, voter_mask: torch.Tensor = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the KemenyTransformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_candidates, dim_input_candidates),
                              representing the base rankings.
            padding_mask (torch.Tensor): A boolean mask for the input tensor `x`.
                                         Shape (batch_size, num_candidates). `True` indicates a padded element.
            deterministic (bool): If True, uses argmax for selection. If False, samples
                                  from the output distribution.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - final_rankings (torch.Tensor): The generated rankings, shape (batch_size, num_candidates).
                                             Contains 1-indexed candidate IDs (1 to num_candidates). Padded steps are 0.
            - avg_log_prob_per_step (torch.Tensor): The average log probability for valid (non-padded)
                                                    actions taken, shape (batch_size,).
            - output_mask (torch.Tensor): A boolean mask for the `final_rankings`.
                                          Shape (batch_size, num_candidates). `True` indicates a padded step.
        """

        bsz = x.shape[0]
        numb_candidates = x.shape[1]
        device = x.device
        zero_to_bsz = torch.arange(bsz, device=device)
        h_encoder, K_att_decoder, V_att_decoder, encoder_positions = self._encode(
            x, padding_mask, voter_mask
        )
        encoder_padding_mask = torch.cat(
            [
                padding_mask.to(device=device, dtype=torch.bool),
                torch.zeros(bsz, 1, device=device, dtype=torch.bool),
            ],
            dim=1,
        )

        # 4. Prepare for Decoder
        # list that will contain Long tensors of shape (bsz,) that gives the idx of the candidates chosen at time t
        final_ranking_steps = []
        # list that will contain Float tensors of shape (bsz,) that gives the neg log probs of the choices made at time t
        sumLogProbOfActions = []
        # list that will contain boolean masks for the output steps
        output_mask_steps = []

        h_t, mask_ranked_candidates, _ = self._prepare_decoder_state(
            h_encoder, encoder_padding_mask
        )

        # 5. Autoregressive Decoding Loop
        # Get the number of valid (non-padded) items for each batch item.
        # We subtract 1 from the sum of the original padding_mask
        # because the mask is True for *padded* items.
        # Shape: (bsz,)
        num_valid_items = (~padding_mask).float().sum(dim=1)

        for t in range(numb_candidates):

            # Combine masks: an item is invalid if it's already been ranked OR if it's padding.
            attention_mask = mask_ranked_candidates | encoder_padding_mask


            # Get probabilities for the next candidate
            query_position = torch.full(
                (bsz,), t, dtype=torch.long, device=device
            )
            pro_next_candidate = self.decoder(
                h_t,
                K_att_decoder,
                V_att_decoder,
                attention_mask,
                query_position=query_position,
                key_positions=encoder_positions,
            )  # size(prob_next_node)=(bsz, num_candidates+1)

            # Choose next candidate (argmax or sampling)
            if deterministic:
                idx = torch.argmax(pro_next_candidate, dim=1)
            else:
                idx = torch.distributions.Categorical(pro_next_candidate).sample() # Shape: (bsz,)

            # compute logprobs of the action items in the list sumLogProbOfActions
            ProbOfChoices = pro_next_candidate[zero_to_bsz, idx]
            sumLogProbOfActions.append(
                torch.log(ProbOfChoices.clamp_min(torch.finfo(pro_next_candidate.dtype).tiny))
            )

            # update embedding of the current visited node
            h_t = self._next_decoder_input(h_encoder, idx, t)

            # is_padded_step is True if t >= num_valid_items for that batch item
            is_padded_step = (t >= num_valid_items) # Shape: (bsz,)

            # Store 0 if padded, otherwise store (idx + 1)
            # We use torch.where to do this efficiently on the batch.
            current_step_ranking = torch.where(
                is_padded_step,
                torch.tensor(0, device=device, dtype=torch.long), # Use 0 for padding
                idx + 1 # Use 1-indexed ID for valid candidates
            )
            final_ranking_steps.append(current_step_ranking)

            # update masks with ranked candidates
            mask_ranked_candidates[zero_to_bsz, idx] = True

        # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
        # Stack all log probabilities: Shape (bsz, num_candidates)
        all_log_probs = torch.stack(sumLogProbOfActions, dim=1)

        # convert the list of candidates in to a tensor of shape (bsz, num_candidates)
        final_rankings = torch.stack(final_ranking_steps, dim=1)

        # Create the final output mask
        # We can derive this directly from the time step logic
        t_range = torch.arange(numb_candidates, device=device).expand(bsz, numb_candidates)
        output_mask = (t_range >= num_valid_items.unsqueeze(1)) # Shape: (bsz, num_candidates)

        # Mask out log-probs from padded steps
        all_log_probs.masked_fill_(output_mask, 0.0)

        # Sum log-probs for valid steps
        sum_log_prob = all_log_probs.sum(dim=1) # Shape: (bsz,)

        # num_valid_steps is just num_valid_items
        num_valid_steps = num_valid_items # Shape: (bsz,)

        # Calculate average, handle division by zero if a batch item has 0 valid steps
        avg_log_prob_per_step = sum_log_prob / num_valid_steps.clamp(min=1.0)

        self.decoder.reset_selfatt_keys_values()
        return final_rankings, avg_log_prob_per_step, output_mask

    def teacher_forcing_forward(
        self,
        noisy_rankings: torch.Tensor,
        target_rankings: torch.Tensor,
        padding_mask: torch.Tensor,
        voter_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute supervised next-item cross entropy for one batch.

        ``target_rankings`` contains zero-based item IDs in ranking order and
        has shape ``[B,N]``.  Invalid target slots may contain any value; they
        are excluded by ``padding_mask``.
        """
        if target_rankings.shape != padding_mask.shape:
            raise ValueError("target_rankings and padding_mask must both be [B,N].")
        bsz, numb_candidates = padding_mask.shape
        target_rankings = target_rankings.to(
            device=noisy_rankings.device, dtype=torch.long
        )
        padding_mask = padding_mask.to(device=noisy_rankings.device, dtype=torch.bool)
        num_valid_items = (~padding_mask).sum(dim=1)
        max_steps = int(num_valid_items.max().item()) if bsz else 0

        h_encoder, K_att_decoder, V_att_decoder, encoder_positions = self._encode(
            noisy_rankings, padding_mask, voter_mask
        )
        encoder_padding_mask = torch.cat(
            [
                padding_mask,
                torch.zeros(bsz, 1, device=noisy_rankings.device, dtype=torch.bool),
            ],
            dim=1,
        )
        h_t, mask_ranked_candidates, _ = self._prepare_decoder_state(
            h_encoder, encoder_padding_mask
        )
        rows = torch.arange(bsz, device=noisy_rankings.device)
        step_log_probs = []
        selected_rankings = []

        for step in range(max_steps):
            valid_step = step < num_valid_items
            current_target = target_rankings[:, step].clamp(
                min=0, max=max(0, numb_candidates - 1)
            )
            if bool((valid_step & (
                (target_rankings[:, step] < 0)
                | (target_rankings[:, step] >= numb_candidates)
            )).any()):
                raise ValueError("A valid target contains an out-of-range item ID.")

            attention_mask = mask_ranked_candidates | encoder_padding_mask
            # The current ground-truth target must remain selectable before this
            # step's loss is calculated.
            target_already_masked = attention_mask.gather(
                1, current_target.unsqueeze(1)
            ).squeeze(1)
            if bool((valid_step & target_already_masked).any()):
                raise ValueError(
                    "target_rankings is not a valid permutation: a current "
                    "target is padded, previously selected, or the START token."
                )

            query_position = torch.full(
                (bsz,), step, dtype=torch.long, device=noisy_rankings.device
            )
            probabilities = self.decoder(
                h_t,
                K_att_decoder,
                V_att_decoder,
                attention_mask,
                query_position=query_position,
                key_positions=encoder_positions,
            )
            log_probabilities = torch.log(
                probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
            )
            step_log_probs.append(
                log_probabilities.gather(1, current_target.unsqueeze(1)).squeeze(1)
            )
            selected_rankings.append(
                torch.where(
                    valid_step,
                    current_target + 1,
                    torch.zeros_like(current_target),
                )
            )

            # Loss has now been formed; only then apply the original mask update.
            mask_ranked_candidates[rows, current_target] = True
            h_t = self._next_decoder_input(h_encoder, current_target, step)

        if step_log_probs:
            all_step_log_probs = torch.stack(step_log_probs, dim=1)
            final_rankings = torch.stack(selected_rankings, dim=1)
        else:
            all_step_log_probs = noisy_rankings.new_zeros((bsz, 0))
            final_rankings = torch.zeros(
                (bsz, 0), dtype=torch.long, device=noisy_rankings.device
            )

        t_range = torch.arange(max_steps, device=noisy_rankings.device).expand(
            bsz, max_steps
        )
        output_mask = t_range >= num_valid_items.unsqueeze(1)
        valid_log_prob = all_step_log_probs.masked_fill(output_mask, 0.0)
        valid_count = (~output_mask).sum()
        if int(valid_count) == 0:
            loss = all_step_log_probs.sum() * 0.0
        else:
            loss = -valid_log_prob.sum() / valid_count.to(dtype=valid_log_prob.dtype)

        self.decoder.reset_selfatt_keys_values()
        return {
            "loss": loss,
            "step_log_probs": all_step_log_probs,
            "rankings": final_rankings,
            "output_mask": output_mask,
        }
