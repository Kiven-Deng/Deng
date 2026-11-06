# v3 fix: support direct script execution; retain v2 model/data behavior and schema.
# 修正说明：
# 1. 共享名次映射 + 对榜单维度做集合均值聚合；模型隐藏维度不再等于榜单数。
# 2. 保留绝对/相对名次、分歧和观测数量；不对原始名次向量直接 LayerNorm。
# 3. 只对学习后的隐藏特征 LayerNorm；移除所有 BatchNorm 运行统计依赖。
# 4. 不使用对象 ID 嵌入；输入榜单列置换不变、对象重编号等变（数值并列除外）。
# 5. teacher forcing 并行因果解码，自由解码屏蔽已选对象；共享完全相同的打分路径。
# 6. rank-log RoPE 与 ANLiBi 风格偏置可分别消融，默认都关闭。
# 7. 支持完整榜单；当前版本不声称支持缺失对象、并列名次或来源身份信息。
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    hidden: int = 64
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    ff_multiplier: int = 4
    rarope: bool = False
    anlibi: bool = False

    def to_dict(self):
        return asdict(self)


class RankAttention(nn.Module):
    """Explicit attention permits separately testable positional ablations.

    Rank-log RoPE uses f(p)=log(1+p), conventional rotary frequencies.
    ANLiBi-style bias = softplus(a)/key_rank - softplus(b)*|f(q)-f(k)|.
    These are v2-defined variants, not a claim of a published implementation.
    """
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.heads, config.hidden // config.heads
        self.rarope, self.anlibi = config.rarope, config.anlibi
        self.query = nn.Linear(config.hidden, config.hidden)
        self.key = nn.Linear(config.hidden, config.hidden)
        self.value = nn.Linear(config.hidden, config.hidden)
        self.output = nn.Linear(config.hidden, config.hidden)
        self.register_buffer("frequencies",
            10000.0 ** (-torch.arange(0, self.width, 2).float() / self.width), persistent=False)
        if self.anlibi:
            initial = math.log(math.expm1(.01))
            self.alpha = nn.Parameter(torch.full((config.heads,), initial))
            self.beta = nn.Parameter(torch.full((config.heads,), initial))

    def rotate(self, value, positions):
        angle = torch.log1p(positions)[:, None, :, None] * self.frequencies
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack((even * angle.cos() - odd * angle.sin(),
                            even * angle.sin() + odd * angle.cos()), -1).flatten(-2)

    def forward(self, query, memory, query_positions, key_positions, blocked=None):
        b, length, hidden = query.shape
        k_length = memory.shape[1]
        q = self.query(query).view(b, length, self.heads, self.width).transpose(1, 2)
        k = self.key(memory).view(b, k_length, self.heads, self.width).transpose(1, 2)
        v = self.value(memory).view(b, k_length, self.heads, self.width).transpose(1, 2)
        if self.rarope:
            q, k = self.rotate(q, query_positions), self.rotate(k, key_positions)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.width)
        if self.anlibi:
            distance = (torch.log1p(query_positions)[:, :, None] -
                        torch.log1p(key_positions)[:, None, :]).abs()
            bias = (F.softplus(self.alpha)[None, :, None, None] /
                    key_positions[:, None, None, :].clamp_min(1) -
                    F.softplus(self.beta)[None, :, None, None] * distance[:, None])
            scores = scores + bias
        if blocked is not None:
            scores = scores.masked_fill(blocked[:, None], float("-inf"))
        attention = scores.softmax(-1)
        return self.output((attention @ v).transpose(1, 2).reshape(b, length, hidden))


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = RankAttention(config)
        self.ff = nn.Sequential(nn.Linear(d, config.ff_multiplier * d), nn.GELU(),
                                nn.Linear(config.ff_multiplier * d, d))

    def forward(self, hidden, positions):
        normalized = self.norm1(hidden)
        hidden = hidden + self.attention(normalized, normalized, positions, positions)
        return hidden + self.ff(self.norm2(hidden))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.memory_norm = nn.LayerNorm(d)
        self.self_attention, self.cross_attention = RankAttention(config), RankAttention(config)
        self.ff = nn.Sequential(nn.Linear(d, config.ff_multiplier * d), nn.GELU(),
                                nn.Linear(config.ff_multiplier * d, d))

    def forward(self, hidden, memory, steps, ranks, causal, selected):
        normalized = self.norm1(hidden)
        hidden = hidden + self.self_attention(normalized, normalized, steps, steps, causal)
        hidden = hidden + self.cross_attention(self.norm2(hidden), self.memory_norm(memory),
                                               steps, ranks, selected)
        return hidden + self.ff(self.norm3(hidden))


class RecoveryTransformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if (c.hidden < 1 or c.heads < 1 or c.hidden % c.heads or
                (c.hidden // c.heads) % 2 or min(c.encoder_layers, c.decoder_layers, c.ff_multiplier) < 1):
            raise ValueError("Need positive layer sizes and an even per-head width.")
        d = c.hidden
        self.observation_map = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.summary_map = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.encoders = nn.ModuleList([EncoderBlock(c) for _ in range(c.encoder_layers)])
        self.decoders = nn.ModuleList([DecoderBlock(c) for _ in range(c.decoder_layers)])
        self.start = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.step_map = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.pointer_query, self.pointer_key = nn.Linear(d, d), nn.Linear(d, d)
        self.pointer_score = nn.Linear(d, 1, bias=False)
        # Explicit scalar-statistic skip preserves access to absolute rank
        # after hidden-state normalization. This is learned, not a Borda fallback.
        self.rank_skip = nn.Linear(6, 1, bias=False)

    def features(self, positions):
        if positions.ndim != 3 or min(positions.shape) < 1:
            raise ValueError("Expected nonempty complete rank matrix [B,N,m].")
        n, m = positions.shape[1:]
        relative = (positions - 1) / max(1, n - 1)
        observation = torch.stack((relative, relative.square(), positions.reciprocal(),
                                   torch.log1p(positions) / math.log1p(n)), dim=-1)
        summary = torch.stack((relative.mean(-1), relative.std(-1, unbiased=False),
                               relative.amin(-1), relative.amax(-1),
                               positions.reciprocal().mean(-1),
                               torch.full_like(relative[:, :, 0], math.log1p(m))), dim=-1)
        return observation, summary

    def encode(self, positions):
        observation, summary = self.features(positions)
        hidden = self.observation_map(observation).mean(2) + self.summary_map(summary)
        ranks = positions.mean(-1)  # Continuous mean ranks: no item-ID tie break.
        for encoder in self.encoders:
            hidden = encoder(hidden, ranks)
        return hidden, ranks, summary

    def prefix_logits(self, memory, ranks, summary, prefix):
        b, n, d = memory.shape
        length = prefix.shape[1] + 1
        selected_memory = memory.gather(1, prefix[:, :, None].expand(-1, -1, d))
        hidden = torch.cat((self.start[None, None].expand(b, 1, -1), selected_memory), 1)
        steps = torch.arange(1, length + 1, device=memory.device, dtype=memory.dtype)[None].expand(b, -1)
        step_features = torch.stack(((steps - 1) / max(1, n - 1),
                                     torch.log1p(steps) / math.log1p(n)), -1)
        hidden = hidden + self.step_map(step_features)
        selected = torch.cat((torch.zeros((b, 1, n), device=memory.device, dtype=torch.bool),
                              F.one_hot(prefix, n).cumsum(1).bool()), 1)
        causal = torch.ones((length, length), device=memory.device, dtype=torch.bool).triu(1)
        causal = causal[None].expand(b, -1, -1)
        for decoder in self.decoders:
            hidden = decoder(hidden, memory, steps, ranks, causal, selected)
        logits = self.pointer_score(torch.tanh(
            self.pointer_query(hidden)[:, :, None, :] +
            self.pointer_key(memory)[:, None, :, :])).squeeze(-1)
        logits = logits + self.rank_skip(summary).transpose(1, 2)
        return logits.masked_fill(selected, float("-inf"))

    def forward(self, positions, targets):
        if targets.shape != positions.shape[:2] or targets.dtype != torch.long:
            raise ValueError("Targets must be zero-based long permutations [B,N].")
        expected = torch.arange(targets.shape[1], device=targets.device).expand_as(targets)
        if not torch.equal(targets.sort(1).values, expected):
            raise ValueError("Targets must be valid permutations.")
        memory, ranks, summary = self.encode(positions)
        return self.prefix_logits(memory, ranks, summary, targets[:, :-1])

    def loss(self, positions, targets):
        logits = self(positions, targets)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    @torch.no_grad()
    def predict(self, positions):
        memory, ranks, summary = self.encode(positions)
        prefix = torch.empty((len(positions), 0), device=positions.device, dtype=torch.long)
        for _ in range(positions.shape[1]):
            logits = self.prefix_logits(memory, ranks, summary, prefix)[:, -1]
            prefix = torch.cat((prefix, logits.argmax(-1)[:, None]), 1)
        return prefix
