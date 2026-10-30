"""Generate synthetic ground-truth rankings and Mallows-noisy observations.

A ranking is stored as an ordered sequence of item IDs.  Values identify the
ranked objects, while their positions in the sequence identify their ranks.
For example, ``[2, 0, 3, 1]`` means item 2 is first, item 0 is second, item 3
is third, and item 1 is fourth.

The model-facing position matrix uses the opposite layout.  Its rows are
indexed by item ID, and its values are one-based ranks.  For the example
above, it is ``[2, 4, 1, 3]``.

To generate a dataset, edit the configuration block below and run this file
directly, for example ``python ranking_data_generator.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

if __package__:
    from .mallows import MallowsPhiSampler
else:
    # This branch makes ``python ranking_data_generator.py`` work without
    # needing PYTHONPATH or ``python -m``.
    from mallows import MallowsPhiSampler


# ============================================================================
# DATASET CONFIGURATION - EDIT ONLY THIS BLOCK
# ============================================================================

# True: create new random ground-truth rankings and overwrite the true file.
# False: reuse TRUE_RANKINGS_FILE.
GENERATE_TRUE_RANKINGS = True

# True: create Mallows-noisy rankings and save the final model dataset.
GENERATE_NOISY_RANKINGS = True

# Number of independent ground-truth ranking tables.
NUM_RANKINGS = 5000

# Number of ranked objects in every table. Item IDs will be 0..NUM_ITEMS-1.
NUM_ITEMS = 50

# Number of noisy ranking tables generated for each true ranking.
NUM_NOISY_RANKINGS = 10

# Mallows concentration: larger values produce less noise; 0 is uniform noise.
MALLOWS_THETA = 2

# Random seeds. Change them to create a different random dataset.
TRUE_RANKINGS_SEED = 42
MALLOWS_SEED = 123

# Relative paths are resolved from the directory containing this file.
TRUE_RANKINGS_FILE = "data/true_rankings.pt"
DATASET_FILE = "data/ranking_dataset.pt"

# ============================================================================
# END OF DATASET CONFIGURATION
# ============================================================================


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RankingDataset:
    """Synthetic ranking data produced around random ground-truth rankings.

    Attributes:
        true_rankings: Item orders with shape ``[num_rankings, num_items]``.
            Item IDs are zero-based, and sequence positions are their ranks.
        noisy_rankings: Mallows samples with shape
            ``[num_rankings, num_noisy_rankings, num_items]``. Each row is a
            complete permutation of item IDs.
        noisy_positions: One-based rank matrices with shape
            ``[num_rankings, num_items, num_noisy_rankings]``. This is the
            float32 layout expected by the identity-embedding denoising model.
    """

    true_rankings: torch.Tensor
    noisy_rankings: torch.Tensor
    noisy_positions: torch.Tensor


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def generate_true_rankings(
    num_rankings: int,
    num_items: int,
    *,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate uniformly random permutations as ground-truth rankings.

    Args:
        num_rankings: Number of independent ranking tables to generate.
        num_items: Number of ranked objects in every table.
        seed: Optional random seed. The same seed and arguments produce the
            same rankings.

    Returns:
        A ``torch.long`` tensor with shape
        ``[num_rankings, num_items]``. Every row contains each ID in
        ``0..num_items-1`` exactly once.
    """
    _validate_positive_integer("num_rankings", num_rankings)
    _validate_positive_integer("num_items", num_items)

    generator = torch.Generator(device="cpu")
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))

    return torch.stack(
        [
            torch.randperm(num_items, generator=generator, dtype=torch.long)
            for _ in range(num_rankings)
        ],
        dim=0,
    )


def generate_noisy_rankings(
    true_rankings: torch.Tensor,
    *,
    num_noisy_rankings: int = 1,
    theta: float = 1.0,
    seed: Optional[int] = None,
) -> RankingDataset:
    """Generate Mallows-noisy rankings around supplied true rankings.

    This is the second stage of dataset construction.  It accepts the output of
    :func:`generate_true_rankings` or a tensor loaded from a true-ranking file.

    Args:
        true_rankings: Ground-truth item orders with shape ``[B,N]``. Values
            must be zero-based IDs and every row must be a permutation of
            ``0..N-1``.
        num_noisy_rankings: Number of noisy rankings generated per true
            ranking. For the identity-embedding model this must equal the
            configured ``embedding_dim``.
        theta: Non-negative Mallows concentration. Larger values produce less
            noise, while ``theta=0`` produces uniform random permutations.
        seed: Optional Mallows sampling seed.

    Returns:
        A :class:`RankingDataset` containing the supplied true rankings, noisy
        item orders, and one-based model input position matrices.
    """
    _validate_positive_integer("num_noisy_rankings", num_noisy_rankings)
    if theta < 0:
        raise ValueError("theta must be non-negative.")

    sampler = MallowsPhiSampler(theta=theta, seed=seed)
    noisy_rankings, _ = sampler.sample_rankings(
        true_rankings,
        num_noisy_rankings,
    )
    noisy_positions = sampler.rankings_to_position_matrix(
        noisy_rankings,
        rank_base=1,
    ).to(dtype=torch.float32)
    true_rankings_cpu = true_rankings.detach().to(
        device="cpu",
        dtype=torch.long,
    )
    return RankingDataset(
        true_rankings=true_rankings_cpu,
        noisy_rankings=noisy_rankings,
        noisy_positions=noisy_positions,
    )


def generate_ranking_dataset(
    num_rankings: int,
    num_items: int,
    *,
    num_noisy_rankings: int = 1,
    theta: float = 1.0,
    seed: Optional[int] = None,
    mallows_seed: Optional[int] = None,
) -> RankingDataset:
    """Generate random true rankings and Mallows-noisy observations.

    Every true ranking is the center of its own Mallows distribution.  A larger
    ``theta`` keeps noisy rankings closer to that center; ``theta=0`` produces
    uniform random permutations.

    Args:
        num_rankings: Number of ground-truth ranking tables.
        num_items: Number of ranked objects in each table.
        num_noisy_rankings: Number of noisy samples generated for every true
            ranking.
        theta: Non-negative Mallows concentration parameter.
        seed: Optional seed for the random ground-truth rankings.
        mallows_seed: Optional seed for Mallows noise. If omitted and ``seed``
            is supplied, a deterministic derived seed is used.

    Returns:
        A :class:`RankingDataset` containing true rankings, noisy rankings,
        and the corresponding one-based position matrices.
    """
    _validate_positive_integer("num_noisy_rankings", num_noisy_rankings)
    if theta < 0:
        raise ValueError("theta must be non-negative.")

    true_rankings = generate_true_rankings(
        num_rankings,
        num_items,
        seed=seed,
    )

    if mallows_seed is None and seed is not None:
        # Samples are reproducible with the dataset seed but do not reuse the
        # same random stream as the center-rank generation.
        mallows_seed = (int(seed) + 1000003) % (2**63)

    return generate_noisy_rankings(
        true_rankings,
        num_noisy_rankings=num_noisy_rankings,
        theta=theta,
        seed=mallows_seed,
    )


def _save_dataset(
    output: Path,
    *,
    true_rankings: torch.Tensor,
    noisy_rankings: Optional[torch.Tensor],
    noisy_positions: Optional[torch.Tensor],
    config: dict,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "true_rankings": true_rankings,
        "config": config,
    }
    if noisy_rankings is not None:
        payload["noisy_rankings"] = noisy_rankings
    if noisy_positions is not None:
        payload["noisy_positions"] = noisy_positions
    torch.save(payload, output)


def _load_true_rankings(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if "true_rankings" not in payload:
            raise ValueError(
                f"{path} does not contain a 'true_rankings' tensor."
            )
        payload = payload["true_rankings"]
    if not isinstance(payload, torch.Tensor) or payload.dim() != 2:
        raise ValueError(
            f"{path} must contain a [num_rankings, num_items] tensor."
        )
    return payload.to(dtype=torch.long)


def _resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else BASE_DIR / resolved


def run_from_config() -> Optional[RankingDataset]:
    """Generate the dataset described by the configuration block."""
    if not GENERATE_TRUE_RANKINGS and not GENERATE_NOISY_RANKINGS:
        raise ValueError(
            "At least one of GENERATE_TRUE_RANKINGS or "
            "GENERATE_NOISY_RANKINGS must be True."
        )

    true_path = _resolve_path(TRUE_RANKINGS_FILE)
    dataset_path = _resolve_path(DATASET_FILE)

    if GENERATE_TRUE_RANKINGS:
        true_rankings = generate_true_rankings(
            NUM_RANKINGS,
            NUM_ITEMS,
            seed=TRUE_RANKINGS_SEED,
        )
        _save_dataset(
            true_path,
            true_rankings=true_rankings,
            noisy_rankings=None,
            noisy_positions=None,
            config={
                "stage": "true",
                "num_rankings": NUM_RANKINGS,
                "num_items": NUM_ITEMS,
                "seed": TRUE_RANKINGS_SEED,
            },
        )
        print(
            f"True rankings saved: {true_path} "
            f"{tuple(true_rankings.shape)}"
        )
    else:
        if not true_path.is_file():
            raise FileNotFoundError(
                f"True-ranking file not found: {true_path}. "
                "Set GENERATE_TRUE_RANKINGS = True first."
            )
        true_rankings = _load_true_rankings(true_path)
        print(
            f"True rankings loaded: {true_path} "
            f"{tuple(true_rankings.shape)}"
        )

    if not GENERATE_NOISY_RANKINGS:
        return None

    dataset = generate_noisy_rankings(
        true_rankings,
        num_noisy_rankings=NUM_NOISY_RANKINGS,
        theta=MALLOWS_THETA,
        seed=MALLOWS_SEED,
    )
    _save_dataset(
        dataset_path,
        true_rankings=dataset.true_rankings,
        noisy_rankings=dataset.noisy_rankings,
        noisy_positions=dataset.noisy_positions,
        config={
            "stage": "both",
            "num_rankings": dataset.true_rankings.size(0),
            "num_items": dataset.true_rankings.size(1),
            "num_noisy_rankings": NUM_NOISY_RANKINGS,
            "theta": MALLOWS_THETA,
            "true_rankings_seed": TRUE_RANKINGS_SEED,
            "mallows_seed": MALLOWS_SEED,
        },
    )
    print(f"Dataset saved: {dataset_path}")
    print(f"  true_rankings:   {tuple(dataset.true_rankings.shape)}")
    print(f"  noisy_rankings:  {tuple(dataset.noisy_rankings.shape)}")
    print(f"  noisy_positions: {tuple(dataset.noisy_positions.shape)}")
    return dataset


if __name__ == "__main__":
    run_from_config()
