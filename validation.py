"""Validate a trained checkpoint on the held-out ranking split.

Edit train_config.py first, then run validation.py directly:

    python validation.py

The script randomly selects VALIDATION_SAMPLE_COUNT rankings from the 10%
validation split, predicts them with no gradient calculation, prints the true
and predicted rankings, and saves the same results as JSON.

Ranking item IDs are zero-based because that is the format used by the
generated dataset. Both rankings are ordered from first place to last place.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Mapping

import torch


if not __package__:
    MODULE_DIR = Path(__file__).resolve().parent
    PACKAGE_PARENT = MODULE_DIR.parent
    if str(PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PARENT))
    __package__ = MODULE_DIR.name

from . import train_config as config
from .ranking_metrics import kendall_tau_distance, rbo_distance, rbo_similarity
from .train import (
    BASE_DIR,
    _select_device,
    _select_num_heads,
    build_model_from_config,
    create_model_config,
    load_ranking_dataset,
    split_dataset_indices,
)


def _resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else BASE_DIR / resolved


def _checkpoint_model_config(
    checkpoint_config: Mapping[str, object],
    num_items: int,
) -> Dict[str, object]:
    """Merge old checkpoint fields with defaults needed by newer models."""
    embedding_dim = int(checkpoint_config["embedding_dim"])
    num_heads = int(
        checkpoint_config.get(
            "numb_heads",
            checkpoint_config.get(
                "num_heads",
                _select_num_heads(embedding_dim, config.NUM_HEADS),
            ),
        )
    )
    dim_ff = int(
        checkpoint_config.get(
            "dim_ff",
            config.DIM_FF if config.DIM_FF is not None else 4 * embedding_dim,
        )
    )
    model_config = create_model_config(
        num_items,
        embedding_dim,
        num_heads,
        dim_ff,
    )
    model_config.update(checkpoint_config)
    model_config["input_dim"] = embedding_dim
    model_config["numb_heads"] = num_heads
    return model_config


def _ranking_list(ranking: torch.Tensor) -> list[int]:
    return [int(item_id) for item_id in ranking.detach().cpu().tolist()]


def validate() -> None:
    if config.VALIDATION_RANDOM_SEED is not None:
        random.seed(config.VALIDATION_RANDOM_SEED)

    device = _select_device(config.DEVICE)
    dataset_path = _resolve_path(config.DATASET_FILE)
    checkpoint_path = _resolve_path(config.CHECKPOINT_FILE)
    output_path = _resolve_path(config.VALIDATION_OUTPUT_FILE)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if config.VALIDATION_SAMPLE_COUNT <= 0:
        raise ValueError("VALIDATION_SAMPLE_COUNT must be positive.")
    if config.VALIDATION_BATCH_SIZE <= 0:
        raise ValueError("VALIDATION_BATCH_SIZE must be positive.")

    noisy_positions, true_rankings = load_ranking_dataset(dataset_path)
    num_samples, num_items, embedding_dim = noisy_positions.shape

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint file must contain a dictionary.")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")

    saved_split = checkpoint.get("dataset_split", {})
    if not isinstance(saved_split, Mapping):
        saved_split = {}
    validation_ratio = float(
        saved_split.get("validation_ratio", config.VALIDATION_RATIO)
    )
    split_seed = int(saved_split.get("split_seed", config.SPLIT_SEED))
    saved_num_samples = saved_split.get("num_samples")
    if saved_num_samples is not None and int(saved_num_samples) != num_samples:
        raise ValueError(
            "The selected dataset does not match the checkpoint split: "
            f"checkpoint samples={saved_num_samples}, current samples={num_samples}."
        )

    _, validation_indices = split_dataset_indices(
        num_samples,
        validation_ratio,
        split_seed,
    )
    validation_size = validation_indices.numel()
    if config.VALIDATION_SAMPLE_COUNT > validation_size:
        raise ValueError(
            f"VALIDATION_SAMPLE_COUNT={config.VALIDATION_SAMPLE_COUNT} exceeds "
            f"the validation set size={validation_size}."
        )

    checkpoint_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Checkpoint does not contain a valid 'model_config'.")
    model_config = _checkpoint_model_config(checkpoint_config, num_items)

    saved_embedding_dim = int(model_config["embedding_dim"])
    if saved_embedding_dim != embedding_dim:
        raise ValueError(
            "Checkpoint and dataset embedding dimensions do not match: "
            f"{saved_embedding_dim} != {embedding_dim}."
        )

    model = build_model_from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    sample_count = config.VALIDATION_SAMPLE_COUNT
    sample_positions = random.sample(range(validation_size), sample_count)
    selected_indices = validation_indices[sample_positions]

    predicted_batches = []
    batch_size = config.VALIDATION_BATCH_SIZE
    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Validation split: {validation_size}/{num_samples} "
        f"({validation_ratio:.2%})"
    )
    print(f"Selected validation rankings: {sample_count}")
    print("Ranking format: ordered item IDs, zero-based")

    for start in range(0, sample_count, batch_size):
        batch_indices = selected_indices[start:start + batch_size]
        model_input = noisy_positions.index_select(0, batch_indices).to(device)
        padding_mask = torch.zeros(
            (batch_indices.numel(), num_items),
            dtype=torch.bool,
            device=device,
        )

        # Validation is inference only: no graph is built and no gradients flow.
        with torch.no_grad():
            predictions_one_based, _, _ = model(
                model_input,
                padding_mask,
                deterministic=True,
            )
        if bool((predictions_one_based == 0).any()):
            raise RuntimeError(
                "The model returned a padded prediction for a full ranking."
            )
        predicted_batches.append(predictions_one_based.cpu() - 1)

    predictions_zero_based = torch.cat(predicted_batches, dim=0)
    results = []
    exact_matches = 0
    kendall_tau_distance_sum = 0.0
    rbo_distance_sum = 0.0

    for position, dataset_index in enumerate(selected_indices.tolist()):
        true_ranking = _ranking_list(true_rankings[dataset_index])
        predicted_ranking = _ranking_list(
            predictions_zero_based[position]
        )
        exact_match = true_ranking == predicted_ranking
        kendall_distance = kendall_tau_distance(
            true_ranking,
            predicted_ranking,
        )
        rbo_similarity_value = rbo_similarity(
            true_ranking,
            predicted_ranking,
            config.RBO_PERSISTENCE,
        )
        rbo_distance_value = rbo_distance(
            true_ranking,
            predicted_ranking,
            config.RBO_PERSISTENCE,
        )
        exact_matches += int(exact_match)
        kendall_tau_distance_sum += kendall_distance
        rbo_distance_sum += rbo_distance_value
        record = {
            "sample": position + 1,
            "dataset_index": int(dataset_index),
            "true_ranking": true_ranking,
            "predicted_ranking": predicted_ranking,
            "exact_match": exact_match,
            "kendall_tau_distance": kendall_distance,
            "rbo_similarity": rbo_similarity_value,
            "rbo_distance": rbo_distance_value,
        }
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))

    output = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "ranking_format": "ordered item IDs, zero-based",
        "validation_ratio": validation_ratio,
        "validation_size": validation_size,
        "sample_count": sample_count,
        "random_seed": config.VALIDATION_RANDOM_SEED,
        "exact_match_count": exact_matches,
        "rbo_persistence": config.RBO_PERSISTENCE,
        "mean_kendall_tau_distance": (
            kendall_tau_distance_sum / sample_count
        ),
        "mean_rbo_distance": rbo_distance_sum / sample_count,
        "predictions": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Exact matches: {exact_matches}/{sample_count}"
    )
    print(
        "Mean Kendall tau distance: "
        f"{kendall_tau_distance_sum / sample_count:.6f}"
    )
    print(
        f"Mean RBO distance: {rbo_distance_sum / sample_count:.6f}"
    )
    print(f"Validation results saved: {output_path}")


if __name__ == "__main__":
    validate()
