"""Independent stagewise generalized Mallows generator (v1).

Run: python generalized_mallows_generator_v1.py
Check: python generalized_mallows_generator_v1.py --self-test

At output stage j (1-based), select the (v_j+1)-th remaining item in CENTER
order, with P(v_j=k) proportional to exp(-theta_j*k), k=0,...,N-j.
Independent codes define the multistage generalized Mallows distribution.
Constant theta recovers Kendall-distance Mallows; theta=0 gives uniform
choices. Large theta stabilizes a selection STAGE, not a fixed item ID.
Items can cross region boundaries. The last stage is forced but its item
can differ from the center's last item. Final displacement is not directly
controlled by theta and need not increase monotonically toward the bottom.
Reference: Tang (2019), section 2.1, equations (2.1)-(2.2):
https://proceedings.mlr.press/v97/tang19a/tang19a.pdf

Only PyTorch and the standard library are required. No old project module
is imported or modified. Outputs use new, exclusive filenames. For future
code changes, create a new version instead of editing this v1 module.
"""
from __future__ import annotations

import argparse
import itertools
import math
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4

import torch

# Illustrative configuration, NOT calibrated to university ranking data.
NUM_RANKINGS = 5000
NUM_ITEMS = 50
NUM_NOISY_RANKINGS = 10
TRUE_RANKINGS_SEED = 42
NOISE_SEED = 123
GENERATION_BATCH_SIZE = 128
# None generates centers; alternatively reuse "data/true_rankings.pt".
# Loaded centers determine B and N, overriding NUM_RANKINGS and NUM_ITEMS.
TRUE_RANKINGS_SOURCE: Optional[str] = None
# Explicit N-1 values take priority over regions below.
THETA_VECTOR: Optional[Sequence[float]] = None
# (inclusive end fraction of N output stages, theta).
# N=50: 1-10 -> 3.0; 11-30 -> 1.0; 31-49 -> 0.35; 50 forced.
THETA_REGIONS = ((0.20, 3.0), (0.60, 1.0), (1.00, 0.35))
OUTPUT_DIRECTORY = "data/generalized_mallows"
BASE_DIR = Path(__file__).resolve().parent


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _theta_values(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError("Every theta must be finite and non-negative.")
    return result


def build_theta_vector(num_items: int, regions=THETA_REGIONS,
                       explicit: Optional[Sequence[float]] = None) -> tuple[float, ...]:
    """Resolve N-1 parameters; region endpoints use ceil(fraction*N)."""
    _positive_int("num_items", num_items)
    if explicit is not None:
        theta = _theta_values(explicit)
        if len(theta) != num_items - 1:
            raise ValueError(f"Expected {num_items - 1} theta values, got {len(theta)}.")
        return theta
    if not regions:
        raise ValueError("regions must not be empty.")
    previous, theta = 0.0, []
    for end_fraction, value in regions:
        end_fraction = float(end_fraction)
        if not math.isfinite(end_fraction) or not previous < end_fraction <= 1:
            raise ValueError("Region endpoints must strictly increase within (0,1].")
        concentration = _theta_values([value])[0]
        stop = min(num_items - 1, math.ceil(end_fraction * num_items))
        theta.extend([concentration] * (stop - len(theta)))
        previous = end_fraction
    if previous != 1.0:
        raise ValueError("The final region endpoint must be 1.0.")
    return tuple(theta)


def _center_tensor(center_rankings) -> torch.Tensor:
    center = torch.as_tensor(center_rankings).detach().cpu()
    if center.ndim != 2 or min(center.shape) < 1:
        raise ValueError("Centers must have nonempty shape [B,N].")
    if center.dtype == torch.bool or center.is_complex():
        raise ValueError("Centers must contain integer item IDs.")
    if center.is_floating_point():
        if not bool(torch.isfinite(center).all()) or not bool((center == center.round()).all()):
            raise ValueError("Centers must contain finite integer item IDs.")
    center = center.to(dtype=torch.long).contiguous()
    expected = torch.arange(center.shape[1]).expand_as(center)
    if not torch.equal(center.sort(dim=1).values, expected):
        raise ValueError("Each center must be a permutation of 0..N-1.")
    return center


@dataclass(frozen=True)
class GeneralizedRankingDataset:
    true_rankings: torch.Tensor  # [B,N], zero-based ordered item IDs
    noisy_rankings: torch.Tensor  # [B,m,N], zero-based ordered item IDs
    noisy_positions: torch.Tensor  # [B,N,m], one-based float32 ranks
    theta_vector: tuple[float, ...]


class GeneralizedMallowsSampler:
    def __init__(self, theta_vector: Sequence[float], seed: Optional[int] = None):
        self.theta_vector = _theta_values(theta_vector)
        self.num_items = len(self.theta_vector) + 1
        self.generator = torch.Generator(device="cpu")
        self.generator.seed() if seed is None else self.generator.manual_seed(seed)
        # Normalize on finite support BEFORE sampling, without tail clipping.
        # Float64 handles zero and tiny theta without inverse-CDF cancellation.
        self.stage_probabilities = tuple(
            torch.softmax(-theta * torch.arange(self.num_items - j, dtype=torch.float64), dim=0)
            for j, theta in enumerate(self.theta_vector)
        )

    def sample_rankings(self, center_rankings, num_rankings: int,
                        batch_size: int = 128) -> torch.Tensor:
        """Return [B,m,N]. Same seed, batching and call order reproduce draws."""
        _positive_int("num_rankings", num_rankings)
        _positive_int("batch_size", batch_size)
        centers = _center_tensor(center_rankings)
        if centers.shape[1] != self.num_items:
            raise ValueError("theta_vector must contain N-1 values.")
        output = torch.empty((len(centers), num_rankings, self.num_items), dtype=torch.long)
        for start in range(0, len(centers), batch_size):
            center_batch = centers[start:start + batch_size]
            count = len(center_batch) * num_rankings
            available = center_batch[:, None, :].expand(-1, num_rankings, -1).reshape(count, self.num_items).clone()
            orders = torch.empty_like(available)
            for j, probabilities in enumerate(self.stage_probabilities):
                choice = torch.multinomial(probabilities, count, replacement=True, generator=self.generator)
                orders[:, j] = available.gather(1, choice[:, None]).squeeze(1)
                # Remove chosen item while retaining center order of the rest.
                indices = torch.arange(available.shape[1] - 1).expand(count, -1)
                indices = indices + (indices >= choice[:, None]).long()
                available = available.gather(1, indices)
            orders[:, -1] = available[:, 0]
            output[start:start + len(center_batch)] = orders.reshape(len(center_batch), num_rankings, self.num_items)
        return output


def generate_true_rankings(num_rankings: int, num_items: int,
                           seed: Optional[int] = None) -> torch.Tensor:
    _positive_int("num_rankings", num_rankings)
    _positive_int("num_items", num_items)
    generator = torch.Generator(device="cpu")
    generator.seed() if seed is None else generator.manual_seed(seed)
    return torch.stack([torch.randperm(num_items, generator=generator) for _ in range(num_rankings)])


def generate_noisy_rankings(true_rankings, *, theta_vector: Sequence[float],
                            num_noisy_rankings: int = 10, seed: Optional[int] = None,
                            batch_size: int = 128) -> GeneralizedRankingDataset:
    centers = _center_tensor(true_rankings)
    sampler = GeneralizedMallowsSampler(theta_vector, seed)
    orders = sampler.sample_rankings(centers, num_noisy_rankings, batch_size)
    ranks = torch.arange(1, centers.shape[1] + 1).expand_as(orders)
    positions = torch.empty_like(orders).scatter_(2, orders, ranks)
    return GeneralizedRankingDataset(centers, orders,
        positions.transpose(1, 2).contiguous().float(), sampler.theta_vector)


def _resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else BASE_DIR / resolved


def run_from_config() -> Path:
    if TRUE_RANKINGS_SOURCE is None:
        centers = generate_true_rankings(NUM_RANKINGS, NUM_ITEMS, TRUE_RANKINGS_SEED)
    else:
        loaded = torch.load(_resolve_path(TRUE_RANKINGS_SOURCE), map_location="cpu", weights_only=True)
        centers = _center_tensor(loaded["true_rankings"] if isinstance(loaded, dict) else loaded)
    theta = build_theta_vector(centers.shape[1], THETA_REGIONS, THETA_VECTOR)
    dataset = generate_noisy_rankings(centers, theta_vector=theta,
        num_noisy_rankings=NUM_NOISY_RANKINGS, seed=NOISE_SEED, batch_size=GENERATION_BATCH_SIZE)
    directory = _resolve_path(OUTPUT_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"generalized_mallows_v1_{stamp}_{uuid4().hex[:8]}.pt"
    payload = {
        "true_rankings": dataset.true_rankings,
        "noisy_rankings": dataset.noisy_rankings,
        "noisy_positions": dataset.noisy_positions,
        "config": {
            "model": "stagewise_generalized_mallows", "generator_version": "v1",
            "num_rankings": len(centers), "num_items": centers.shape[1],
            "num_noisy_rankings": NUM_NOISY_RANKINGS, "theta_vector": list(theta),
            "theta_regions": list(THETA_REGIONS) if THETA_VECTOR is None else None,
            "theta_semantics": "output selection stage; last stage forced",
            "true_rankings_source": str(_resolve_path(TRUE_RANKINGS_SOURCE)) if TRUE_RANKINGS_SOURCE else None,
            "true_rankings_seed": TRUE_RANKINGS_SEED if TRUE_RANKINGS_SOURCE is None else None,
            "noise_seed": NOISE_SEED, "generation_batch_size": GENERATION_BATCH_SIZE,
            "torch_version": str(torch.__version__),
        },
    }
    with path.open("xb") as stream:  # Never overwrite an existing file.
        torch.save(payload, stream)
    print(f"Saved new dataset: {path}")
    for key in ("true_rankings", "noisy_rankings", "noisy_positions"):
        print(f"  {key}: {tuple(payload[key].shape)}")
    print(f"  theta_vector: {list(theta)}")
    return path


def self_test() -> None:
    """Check small-N laws, empirical sampling, reproducibility and layouts."""
    torch.set_num_threads(1)
    assert build_theta_vector(50) == (3.0,) * 10 + (1.0,) * 20 + (0.35,) * 19
    for theta in ([0.0] * 3, [0.7] * 3, [2.0, 0.6, 0.1]):
        sampler = GeneralizedMallowsSampler(theta, seed=321)
        permutations = list(itertools.permutations(range(4)))
        probabilities, codes = [], []
        for permutation in permutations:
            remaining, stage_codes = list(range(4)), []
            for item in permutation[:-1]:
                stage_codes.append(remaining.index(item))
                remaining.remove(item)
            codes.append(stage_codes)
            probabilities.append(math.prod(float(sampler.stage_probabilities[j][k]) for j, k in enumerate(stage_codes)))
        expected = torch.softmax(-torch.tensor(codes, dtype=torch.float64).matmul(torch.tensor(theta, dtype=torch.float64)), dim=0)
        assert torch.allclose(torch.tensor(probabilities, dtype=torch.float64), expected, atol=1e-14, rtol=1e-12)
        draws = sampler.sample_rankings(torch.arange(4)[None], 60000)[0]
        observed = torch.tensor([(draws == torch.tensor(p)).all(dim=1).float().mean() for p in permutations])
        tolerance = 6 * torch.sqrt(expected * (1 - expected) / len(draws)) + 2 / len(draws)
        assert bool((abs(observed - expected) <= tolerance).all()), "Empirical law mismatch"
    sampler = GeneralizedMallowsSampler([0.7], 42)
    assert abs(float(sampler.stage_probabilities[0][1]) - math.exp(-0.7) / (1 + math.exp(-0.7))) < 1e-14
    centers = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
    kwargs = dict(theta_vector=[2, 0.6, 0.1], num_noisy_rankings=8, seed=11, batch_size=1)
    result = generate_noisy_rankings(centers, **kwargs)
    repeat = generate_noisy_rankings(centers, **kwargs)
    assert torch.equal(result.noisy_rankings, repeat.noisy_rankings)
    assert torch.equal(result.noisy_rankings.sort(dim=-1).values, torch.arange(4).expand(2, 8, 4))
    assert torch.equal(result.noisy_positions.transpose(1, 2).long().gather(2, result.noisy_rankings), torch.arange(1, 5).expand(2, 8, 4))
    assert result.noisy_positions.dtype == torch.float32 and result.noisy_positions.shape == (2, 4, 8)
    stable = GeneralizedMallowsSampler([1000] * 3, 1).sample_rankings(centers, 4)
    assert torch.equal(stable, centers[:, None].expand(-1, 4, -1))
    singleton = GeneralizedMallowsSampler([], 1).sample_rankings(torch.tensor([[0]]), 3)
    assert singleton.shape == (1, 3, 1) and not bool(singleton.any())
    assert torch.allclose(GeneralizedMallowsSampler([1e-15]).stage_probabilities[0], torch.tensor([0.5, 0.5], dtype=torch.float64))
    for invalid in ([-1.0], [float("nan")], [float("inf")]):
        try:
            GeneralizedMallowsSampler(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid theta accepted")
    with tempfile.TemporaryDirectory(prefix="generalized_mallows_test_") as directory:
        path = Path(directory) / "roundtrip.pt"
        torch.save({"true_rankings": result.true_rankings, "noisy_rankings": result.noisy_rankings,
                    "noisy_positions": result.noisy_positions}, path)
        loaded = torch.load(path, weights_only=True)
        assert torch.equal(loaded["noisy_positions"], result.noisy_positions)
    print("PASS: normalized PMFs, exhaustive N=4 law, empirical sampling, reproducibility, layouts, limits, validation, .pt roundtrip.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Check sampler without generating datasets")
    args = parser.parse_args()
    self_test() if args.self_test else run_from_config()
