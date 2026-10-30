"""
Kemeny Transformer Inference Script.

Runs greedy inference on a dataset using a trained checkpoint,
computes Kemeny distances, and saves results.

Usage:
  python scripts/inference.py \
    --config configs/args_8_voter_100_items.conf \
    --checkpoint path/to/checkpoint.pkl \
    --dataset path/to/dataset.npy \
    --output-dir results/ \
    --output-name test_dataset_random

  # Custom batch size and GPU:
  python scripts/inference.py \
    --config configs/args_8_voter_100_items.conf \
    --checkpoint path/to/checkpoint.pkl \
    --dataset path/to/dataset.npy \
    --output-dir results/ \
    --output-name test_dataset_random \
    --batch-size 64 \
    --gpu-id 0
"""
import argparse
import json
import os
from time import time

import numpy as np
import torch

from kemeny_transformer.model.architecture import kemeny_transformer as KemenyTransformerModel
from kemeny_transformer.model.architecture import EmbeddingType, DotDict
from kemeny_transformer.model.tokenization import KemenyTransformerTokenization
from kemeny_transformer.utils.kemeny_distance import (
    compute_kemeny_distance_parallel_greedy,
    permutation_to_ranking_greedy,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Kemeny Transformer Inference')
    parser.add_argument('--config', required=True, help='Path to model config file (.conf)')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pkl)')
    parser.add_argument('--dataset', required=True,
                        help='Path to dataset (.npy). Shape: (num_samples, num_voters, num_candidates)')
    parser.add_argument('--output-dir', required=True, help='Directory to save results')
    parser.add_argument('--output-name', required=True, help='Output file name (without extension)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for inference (default: 128)')
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='GPU ID to use (default: from config)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples to process')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def build_model_args(config, gpu_id_override=None):
    args = DotDict()
    args.dim_input = config['dim_input']
    args.dim_emb = config['dim_emb']
    args.dim_ff = config['dim_ff']
    args.numb_heads = config['numb_heads']
    args.numb_layers_decoder = config['numb_layers_decoder']
    args.numb_layers_encoder = config['numb_layers_encoder']
    args.max_len_PE = config['max_len_PE']
    args.batchnorm = config['batchnorm']
    args.gpu_id = gpu_id_override if gpu_id_override is not None else config['gpu_id']
    args.conv_out_channels = config.get('conv_out_channels', 64)
    args.normalize_input = config.get('normalize_input', 'False') == 'True'

    embedding_type_str = config.get('embedding_type', 'linear')
    if embedding_type_str.lower() == 'linear':
        args.embedding_type = EmbeddingType.LINEAR
    elif embedding_type_str.lower() == 'conv':
        args.embedding_type = EmbeddingType.CONV
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type_str}")

    return args


def load_model(args, checkpoint_path, device):
    model = KemenyTransformerModel(
        embedding_type=args.embedding_type,
        input_dim=args.dim_input,
        embedding_dim=args.dim_emb,
        dim_ff=args.dim_ff,
        numb_heads=args.numb_heads,
        numb_layers_decoder=args.numb_layers_decoder,
        numb_layers_encoder=args.numb_layers_encoder,
        max_len_PE=args.max_len_PE,
        conv_out_channels=args.conv_out_channels,
        batchnorm=args.batchnorm
    )
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DDP-wrapped state dicts (keys prefixed with 'module.')
    def strip_ddp_prefix(state_dict):
        return {k.replace('module.', '', 1) if k.startswith('module.') else k: v
                for k, v in state_dict.items()}

    if 'model_baseline_state_dict' in checkpoint:
        state_dict = strip_ddp_prefix(checkpoint['model_baseline_state_dict'])
        model.load_state_dict(state_dict)
        print(f"  Loaded weights from 'model_baseline_state_dict'")
    elif 'model_train_state_dict' in checkpoint:
        state_dict = strip_ddp_prefix(checkpoint['model_train_state_dict'])
        model.load_state_dict(state_dict)
        print(f"  Loaded weights from 'model_train_state_dict'")
    else:
        raise ValueError(f"Could not find model state dict. Keys: {list(checkpoint.keys())}")

    epoch = checkpoint.get('epoch', -1) + 1
    print(f"  Checkpoint epoch: {epoch}")

    del checkpoint
    model.eval()
    return model


def run_greedy(model, tokenizer, dataset, batch_size, embedding_type, device):
    """Run greedy (deterministic) inference on the dataset."""
    dataset_size = dataset.shape[0]
    all_rankings = []
    all_log_probs = []

    with torch.no_grad():
        start_time = time()
        for batch_start in range(0, dataset_size, batch_size):
            batch_end = min(batch_start + batch_size, dataset_size)
            batch_data = dataset[batch_start:batch_end]
            batch_list = [batch_data[i] for i in range(len(batch_data))]

            padded_batch, padding_mask, voter_mask = tokenizer.tokenize(
                batch_list, embedding_type=embedding_type
            )
            padded_batch = padded_batch.to(device)
            padding_mask = padding_mask.to(device)
            voter_mask = voter_mask.to(device)

            final_rankings, avg_log_probs, output_mask = model(
                padded_batch, padding_mask, voter_mask=voter_mask, deterministic=True
            )

            all_rankings.append(final_rankings.cpu())
            all_log_probs.append(avg_log_probs.cpu())

            print(f"  Greedy batch {batch_start//batch_size + 1}/"
                  f"{(dataset_size + batch_size - 1)//batch_size}")

        elapsed = time() - start_time

    all_rankings = torch.cat(all_rankings, dim=0).numpy()
    all_log_probs = torch.cat(all_log_probs, dim=0).numpy()

    print(f"  Greedy inference: {elapsed:.2f}s ({elapsed/dataset_size:.4f}s/sample)")
    return all_rankings, all_log_probs, elapsed


def main():
    args = parse_args()

    # Load config and build model args
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}/{args.output_name}_transformer.npy")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*80}")

    config = load_config(args.config)
    model_args = build_model_args(config, gpu_id_override=args.gpu_id)

    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_args.gpu_id)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} (ID: {model_args.gpu_id})")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load model
    print("\nLoading model...")
    model = load_model(model_args, args.checkpoint, device)

    # Initialize tokenizer
    tokenizer = KemenyTransformerTokenization(
        max_voters=model_args.dim_input,
        pad_value=0.0,
        normalize_input=model_args.normalize_input
    )

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = np.load(args.dataset)
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]
    print(f"  Shape: {dataset.shape} "
          f"({dataset.shape[0]} samples, {dataset.shape[1]} voters, {dataset.shape[2]} candidates)")

    # Run greedy inference
    results = {}
    data_type = args.output_name

    print(f"\n{'='*80}")
    print("Running greedy inference...")
    print(f"{'='*80}")
    greedy_rankings, greedy_log_probs, greedy_time = run_greedy(
        model, tokenizer, dataset, args.batch_size, model_args.embedding_type, device
    )
    results[f'final ranking greedy {data_type} permutation'] = greedy_rankings
    results[f'greedy time {data_type}'] = greedy_time
    results[f'greedy log probs {data_type}'] = greedy_log_probs

    # Compute Kemeny distances
    print(f"\n{'='*80}")
    print("Computing Kemeny distances...")
    print(f"{'='*80}")

    greedy_rankings_converted = permutation_to_ranking_greedy(greedy_rankings)
    greedy_kemeny_distances = compute_kemeny_distance_parallel_greedy(dataset, greedy_rankings_converted)
    results[f'greedy kemeny distance {data_type}'] = greedy_kemeny_distances
    results[f'greedy kemeny distance mean {data_type}'] = np.nanmean(greedy_kemeny_distances)
    print(f"  Greedy Kemeny distance mean: {np.nanmean(greedy_kemeny_distances):.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_name}_transformer.npy")
    np.save(output_path, results)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"Keys: {list(results.keys())}")
    print(f"  Greedy time: {greedy_time:.2f}s")
    print(f"  Greedy Kemeny distance mean: {results[f'greedy kemeny distance mean {data_type}']:.4f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
