"""Train the Mallows ranking-denoising Transformer from a generated dataset.

Edit train_config.py, then run this file directly:

    python train.py

The dataset must contain:
    noisy_positions: float tensor [num_samples, num_items, num_noisy_rankings]
    true_rankings:   long tensor  [num_samples, num_items]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


if not __package__:
    # Running train.py directly does not put the project root on sys.path.
    # Initialize its package context so relative imports work in both modes.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    __package__ = "kemeny_transformer.model"

from . import train_config as config
from .architecture import kemeny_transformer


BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else BASE_DIR / resolved


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config.DEVICE is 'cuda', but CUDA is not available.")
    return device


def _select_num_heads(embedding_dim: int, requested: Optional[int]) -> int:
    if requested is not None:
        if embedding_dim % requested != 0:
            raise ValueError(
                f"NUM_HEADS={requested} must divide embedding_dim="
                f"{embedding_dim}."
            )
        if (embedding_dim // requested) % 2 != 0:
            raise ValueError(
                "RARoPE requires an even dimension per attention head: "
                f"embedding_dim={embedding_dim}, NUM_HEADS={requested}."
            )
        return requested

    candidates = [
        heads
        for heads in range(1, min(8, embedding_dim) + 1)
        if embedding_dim % heads == 0
        and (embedding_dim // heads) % 2 == 0
    ]
    if candidates:
        return max(candidates)
    raise ValueError(
        f"No valid attention-head count found for embedding_dim={embedding_dim}."
    )


def load_ranking_dataset(
    path: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and validate the Encoder input and Decoder target tensors."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Dataset file must contain a dictionary.")
    if "noisy_positions" not in payload or "true_rankings" not in payload:
        raise ValueError(
            "Dataset must contain 'noisy_positions' and 'true_rankings'."
        )

    noisy_positions = payload["noisy_positions"]
    true_rankings = payload["true_rankings"]
    if not isinstance(noisy_positions, torch.Tensor):
        raise TypeError("noisy_positions must be a torch.Tensor.")
    if not isinstance(true_rankings, torch.Tensor):
        raise TypeError("true_rankings must be a torch.Tensor.")
    if noisy_positions.dim() != 3:
        raise ValueError(
            "noisy_positions must have shape [samples, items, noisy_rankings]."
        )
    if true_rankings.dim() != 2:
        raise ValueError("true_rankings must have shape [samples, items].")
    if noisy_positions.shape[:2] != true_rankings.shape:
        raise ValueError(
            "noisy_positions and true_rankings must agree on "
            f"[samples, items]: got {tuple(noisy_positions.shape)} and "
            f"{tuple(true_rankings.shape)}."
        )

    noisy_positions = noisy_positions.to(dtype=torch.float32).contiguous()
    true_rankings = true_rankings.to(dtype=torch.long).contiguous()
    return noisy_positions, true_rankings


def build_dataloader(
    noisy_positions: torch.Tensor,
    true_rankings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("BATCH_SIZE must be positive.")
    dataset = TensorDataset(noisy_positions, true_rankings)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=config.SHUFFLE,
        num_workers=config.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=config.DROP_LAST,
    )


def split_dataset_indices(
    num_samples: int,
    validation_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic, disjoint train and validation index tensors."""
    if num_samples < 2:
        raise ValueError("At least two samples are needed for a split.")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("VALIDATION_RATIO must be between 0 and 1.")

    validation_size = max(1, int(round(num_samples * validation_ratio)))
    validation_size = min(validation_size, num_samples - 1)
    train_size = num_samples - validation_size

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shuffled = torch.randperm(num_samples, generator=generator)
    return shuffled[:train_size], shuffled[train_size:]


def create_model_config(
    num_items: int,
    embedding_dim: int,
    num_heads: int,
    dim_ff: int,
) -> Dict[str, object]:
    """Return every setting required to reconstruct the trained model."""
    return {
        "embedding_type": "mallows_identity",
        "input_dim": embedding_dim,
        "embedding_dim": embedding_dim,
        "dim_ff": dim_ff,
        "numb_heads": num_heads,
        "numb_layers_decoder": config.NUM_LAYERS_DECODER,
        "numb_layers_encoder": config.NUM_LAYERS_ENCODER,
        "max_len_PE": max(config.MAX_LEN_PE, num_items + 2),
        "num_generated_rankings": embedding_dim,
        "mallows_theta": config.MALLOWS_THETA,
        "mallows_seed": config.MALLOWS_SEED,
        "mallows_resample_each_epoch": True,
        "norm_type": config.NORM_TYPE,
        "norm_position": config.NORM_POSITION,
        "use_rarope": config.USE_RAROPE,
        "use_anlibi": config.USE_ANLIBI,
        "rarope_a": config.RAROPE_A,
        "rarope_b": config.RAROPE_B,
        "anlibi_alpha_init": config.ANLIBI_ALPHA_INIT,
        "anlibi_beta_init": config.ANLIBI_BETA_INIT,
        "anlibi_gamma": config.ANLIBI_GAMMA,
        "use_original_decoder_pe": config.USE_ORIGINAL_DECODER_PE,
        "use_rarope_encoder": config.USE_RAROPE_ENCODER,
        "use_rarope_decoder_self": config.USE_RAROPE_DECODER_SELF,
        "use_rarope_decoder_cross": config.USE_RAROPE_DECODER_CROSS,
        "use_anlibi_encoder": config.USE_ANLIBI_ENCODER,
        "use_anlibi_decoder_self": config.USE_ANLIBI_DECODER_SELF,
        "use_anlibi_decoder_cross": config.USE_ANLIBI_DECODER_CROSS,
        "training_mode": config.TRAINING_MODE,
        "loss_type": config.LOSS_TYPE,
        "preserve_original_mask_update": (
            config.PRESERVE_ORIGINAL_MASK_UPDATE
        ),
    }


def build_model_from_config(
    model_config: Mapping[str, object],
) -> kemeny_transformer:
    """Construct a model from a saved or generated model configuration."""
    mallows_seed = model_config.get("mallows_seed")
    if mallows_seed is not None:
        mallows_seed = int(mallows_seed)

    return kemeny_transformer(
        embedding_type=str(model_config["embedding_type"]),
        input_dim=int(model_config["input_dim"]),
        embedding_dim=int(model_config["embedding_dim"]),
        dim_ff=int(model_config["dim_ff"]),
        numb_heads=int(model_config["numb_heads"]),
        numb_layers_decoder=int(model_config["numb_layers_decoder"]),
        numb_layers_encoder=int(model_config["numb_layers_encoder"]),
        max_len_PE=int(model_config["max_len_PE"]),
        num_generated_rankings=int(model_config["num_generated_rankings"]),
        mallows_theta=float(model_config["mallows_theta"]),
        mallows_seed=mallows_seed,
        mallows_resample_each_epoch=bool(
            model_config["mallows_resample_each_epoch"]
        ),
        norm_type=str(model_config["norm_type"]),
        norm_position=str(model_config["norm_position"]),
        use_rarope=bool(model_config["use_rarope"]),
        use_anlibi=bool(model_config["use_anlibi"]),
        rarope_a=float(model_config["rarope_a"]),
        rarope_b=float(model_config["rarope_b"]),
        anlibi_alpha_init=float(model_config["anlibi_alpha_init"]),
        anlibi_beta_init=float(model_config["anlibi_beta_init"]),
        anlibi_gamma=float(model_config["anlibi_gamma"]),
        use_original_decoder_pe=bool(
            model_config["use_original_decoder_pe"]
        ),
        use_rarope_encoder=_optional_bool(
            model_config["use_rarope_encoder"]
        ),
        use_rarope_decoder_self=_optional_bool(
            model_config["use_rarope_decoder_self"]
        ),
        use_rarope_decoder_cross=_optional_bool(
            model_config["use_rarope_decoder_cross"]
        ),
        use_anlibi_encoder=_optional_bool(
            model_config["use_anlibi_encoder"]
        ),
        use_anlibi_decoder_self=_optional_bool(
            model_config["use_anlibi_decoder_self"]
        ),
        use_anlibi_decoder_cross=_optional_bool(
            model_config["use_anlibi_decoder_cross"]
        ),
        training_mode=str(model_config["training_mode"]),
        loss_type=str(model_config["loss_type"]),
        preserve_original_mask_update=bool(
            model_config["preserve_original_mask_update"]
        ),
    )


def _optional_bool(value: object) -> Optional[bool]:
    return None if value is None else bool(value)


def build_model(num_items: int, embedding_dim: int) -> kemeny_transformer:
    num_heads = _select_num_heads(embedding_dim, config.NUM_HEADS)
    dim_ff = (
        config.DIM_FF
        if config.DIM_FF is not None
        else 4 * embedding_dim
    )
    return build_model_from_config(
        create_model_config(num_items, embedding_dim, num_heads, dim_ff)
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: Mapping[str, object],
    dataset_split: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": model_config,
            "dataset_split": dataset_split,
        },
        path,
    )


def train(max_batches: Optional[int] = None) -> None:
    """Train one or more epochs using fixed-size DataLoader batches."""
    if config.NUM_EPOCHS <= 0:
        raise ValueError("config.NUM_EPOCHS must be positive.")
    if config.LEARNING_RATE <= 0:
        raise ValueError("config.LEARNING_RATE must be positive.")
    if config.CHECKPOINT_SAVE_EVERY_N_EPOCHS <= 0:
        raise ValueError(
            "config.CHECKPOINT_SAVE_EVERY_N_EPOCHS must be positive."
        )

    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    device = _select_device(config.DEVICE)
    dataset_path = _resolve_path(config.DATASET_FILE)
    checkpoint_path = _resolve_path(config.CHECKPOINT_FILE)
    noisy_positions, true_rankings = load_ranking_dataset(dataset_path)

    num_samples, num_items, embedding_dim = noisy_positions.shape
    train_indices, validation_indices = split_dataset_indices(
        num_samples,
        config.VALIDATION_RATIO,
        config.SPLIT_SEED,
    )
    train_noisy_positions = noisy_positions.index_select(0, train_indices)
    train_true_rankings = true_rankings.index_select(0, train_indices)
    if (
        config.EMBEDDING_DIM is not None
        and config.EMBEDDING_DIM != embedding_dim
    ):
        raise ValueError(
            f"config.EMBEDDING_DIM={config.EMBEDDING_DIM} does not match "
            f"dataset d={embedding_dim}."
        )
    num_heads = _select_num_heads(embedding_dim, config.NUM_HEADS)
    dim_ff = (
        config.DIM_FF
        if config.DIM_FF is not None
        else 4 * embedding_dim
    )

    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {num_samples}")
    print(f"Items per ranking: {num_items}")
    print(f"Noisy rankings per sample: {embedding_dim}")
    print(
        f"Split: train={train_indices.numel()}, "
        f"validation={validation_indices.numel()} "
        f"(validation ratio={config.VALIDATION_RATIO:.2%})"
    )
    print(f"Batch size: {config.BATCH_SIZE}")
    print(
        f"Model: embedding_dim={embedding_dim}, heads={num_heads}, "
        f"dim_ff={dim_ff}, encoder_layers={config.NUM_LAYERS_ENCODER}, "
        f"decoder_layers={config.NUM_LAYERS_DECODER}"
    )

    loader = build_dataloader(
        train_noisy_positions,
        train_true_rankings,
        config.BATCH_SIZE,
        device,
    )
    model_config = create_model_config(
        num_items,
        embedding_dim,
        num_heads,
        dim_ff,
    )
    model = build_model_from_config(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    if max_batches is None:
        max_batches = config.MAX_BATCHES

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss_sum = 0.0
        processed_batches = 0
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}",
            unit="batch",
            dynamic_ncols=True,
        )

        for noisy_batch, true_batch in progress:
            optimizer.zero_grad(set_to_none=True)
            output = model.training_forward_from_dataset(
                noisy_batch,
                true_batch,
            )

            # Teacher Forcing returns mean next-item Cross-Entropy.
            loss = output["loss"]
            loss.backward()
            if config.GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.GRAD_CLIP_NORM,
                )
            optimizer.step()

            batch_loss = float(loss.detach().item())
            epoch_loss_sum += batch_loss
            processed_batches += 1
            progress.set_postfix(
                loss=f"{batch_loss:.4f}",
                avg_loss=f"{epoch_loss_sum / processed_batches:.4f}",
            )

            if max_batches is not None and processed_batches >= max_batches:
                break

        progress.close()
        if processed_batches == 0:
            raise RuntimeError("The DataLoader produced no training batches.")

        avg_epoch_loss = epoch_loss_sum / processed_batches
        print(
            f"Epoch {epoch + 1} complete | "
            f"average cross-entropy={avg_epoch_loss:.6f}"
        )

        should_save = (
            config.SAVE_CHECKPOINT
            and (epoch + 1) % config.CHECKPOINT_SAVE_EVERY_N_EPOCHS == 0
        )
        if should_save:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                model_config,
                {
                    "dataset_file": str(dataset_path),
                    "num_samples": num_samples,
                    "train_size": train_indices.numel(),
                    "validation_size": validation_indices.numel(),
                    "validation_ratio": config.VALIDATION_RATIO,
                    "split_seed": config.SPLIT_SEED,
                },
            )
            print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    train()
