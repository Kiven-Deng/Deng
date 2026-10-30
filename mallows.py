"""Exact Mallows-phi noise generation for ranking-denoising inputs.

The sampler uses the inversion table (Lehmer code) representation.  For a
permutation relative to the identity, every inversion-table entry y_i follows
an independent truncated geometric distribution

    P(y_i = k) proportional to exp(-theta * k), 0 <= k <= N - i - 1.

Since Kendall tau is the sum of the inversion-table entries, this produces an
exact sample from the one-parameter Mallows phi model.  Composing the sampled
relative permutation with the center ranking preserves Kendall distance.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch


TensorLike = Any


class MallowsPhiSampler:
    """Generate independent Mallows-phi rankings around supplied center ranks.

    Center and output rankings use zero-based item IDs.  The model-facing
    position matrix uses one-based ranks (winning item has rank 1).

    Args:
        theta: Mallows concentration.  Must be non-negative; larger values put
            more mass near the center ranking.
        seed: Optional CPU random seed.  A fixed seed and identical call order
            make sampling reproducible.
    """

    def __init__(self, theta: float = 1.0, seed: Optional[int] = None):
        if theta < 0:
            raise ValueError("Mallows theta must be non-negative.")
        self.theta = float(theta)
        self.generator = torch.Generator(device="cpu")
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(int(seed))

    @staticmethod
    def _as_rankings(center_rankings: TensorLike) -> torch.Tensor:
        if isinstance(center_rankings, torch.Tensor):
            center = center_rankings.detach().to(device="cpu", dtype=torch.long)
        else:
            # Import only when a NumPy array is actually supplied.
            import numpy as np

            center = torch.as_tensor(np.asarray(center_rankings), dtype=torch.long)

        if center.dim() != 2:
            raise ValueError(
                "center_rankings must have shape [batch, num_items]."
            )
        num_items = center.size(1)
        expected = torch.arange(num_items).unsqueeze(0)
        valid = torch.sort(center, dim=1).values.eq(expected).all()
        if not bool(valid):
            raise ValueError(
                "Every center ranking must be a permutation of 0..N-1."
            )
        return center

    def _sample_relative_permutations(
        self,
        batch_size: int,
        num_rankings: int,
        num_items: int,
    ) -> torch.Tensor:
        """Sample permutations relative to the identity, shape [B, d, N]."""
        uniforms = torch.rand(
            (batch_size, num_rankings, num_items),
            generator=self.generator,
            dtype=torch.float32,
        )
        max_inversion = (
            num_items - 1 - torch.arange(num_items, dtype=torch.long)
        )
        if self.theta < 1.0e-7:
            # The theta=0 Mallows model is the uniform distribution.  Sampling
            # the allowed inversion index directly avoids an unstable q=1 log.
            inversion_table = torch.floor(
                uniforms
                * (max_inversion.to(dtype=torch.float32) + 1.0)
            ).to(dtype=torch.long)
        else:
            # For truncated geometric Y with P(Y=k) proportional to q^k:
            #   Y = min(floor(log(1-U) / log(q)), N-i-1).
            # This is an exact inverse-CDF transform with O(B*d*N) memory.
            q = math.exp(-self.theta)
            inversion_table = torch.floor(
                torch.log1p(-uniforms) / math.log(q)
            ).to(dtype=torch.long)
        inversion_table = torch.minimum(inversion_table, max_inversion)

        # Decode Lehmer codes left-to-right.  At position i, select the element
        # at index y_i from the sorted set of elements not selected yet.
        available = torch.arange(num_items, dtype=torch.long).expand(
            batch_size, num_rankings, num_items
        ).clone()
        output = torch.empty_like(available)
        for position in range(num_items):
            is_available = available >= 0
            available_index = is_available.cumsum(dim=-1) - 1
            take = is_available & (
                available_index == inversion_table[..., position].unsqueeze(-1)
            )
            selected = (available.masked_fill(~take, 0)).sum(dim=-1)
            output[..., position] = selected
            available.masked_fill_(take, -1)
        return output

    def sample_rankings(
        self,
        center_rankings: TensorLike,
        num_rankings: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Sample rankings and return ``orders [B,d,N], d``.

        This overload returns two values so callers can assert that the number
        of generated rankings equals the encoder hidden dimension.
        """
        center = self._as_rankings(center_rankings)
        batch_size, num_items = center.shape
        if num_rankings is None:
            raise ValueError("num_rankings must be supplied explicitly.")
        if num_rankings <= 0:
            raise ValueError("num_rankings must be positive.")

        relative = self._sample_relative_permutations(
            batch_size, int(num_rankings), num_items
        )
        noisy_orders = center.unsqueeze(1).expand(
            -1, int(num_rankings), -1
        ).gather(-1, relative)
        return noisy_orders, int(num_rankings)

    @staticmethod
    def rankings_to_position_matrix(
        rankings: torch.Tensor,
        rank_base: int = 1,
    ) -> torch.Tensor:
        """Convert item orders ``[B,d,N]`` to item ranks ``[B,N,d]``."""
        if rankings.dim() != 3:
            raise ValueError("rankings must have shape [batch, d, num_items].")
        if rank_base not in (0, 1):
            raise ValueError("rank_base must be 0 or 1.")
        batch_size, num_rankings, num_items = rankings.shape
        ranks = (torch.arange(num_items, dtype=torch.long) + rank_base).expand(
            batch_size, num_rankings, num_items
        )
        positions = torch.zeros_like(rankings)
        positions.scatter_(-1, rankings, ranks)
        return positions.transpose(1, 2).contiguous()

    def sample_position_matrix(
        self,
        center_rankings: TensorLike,
        num_rankings: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate model inputs directly as ``[B,N,d]`` one-based ranks."""
        orders, _ = self.sample_rankings(center_rankings, num_rankings)
        return self.rankings_to_position_matrix(orders, rank_base=1)

    @staticmethod
    def kendall_distance(
        ranking_a: TensorLike,
        ranking_b: TensorLike,
    ) -> torch.Tensor:
        """Return pairwise Kendall distances for rankings ``[B,N]``."""
        a = MallowsPhiSampler._as_rankings(ranking_a)
        b = MallowsPhiSampler._as_rankings(ranking_b)
        if a.shape != b.shape:
            raise ValueError("rankings must have the same shape.")

        position_b = torch.empty_like(b)
        item_ids = torch.arange(b.size(1)).expand_as(b)
        position_b.scatter_(1, b, item_ids)
        a_order = position_b.gather(1, a)
        pair_i, pair_j = torch.triu_indices(
            a.size(1), a.size(1), offset=1
        )
        return (a_order[:, pair_i] > a_order[:, pair_j]).sum(dim=1)
