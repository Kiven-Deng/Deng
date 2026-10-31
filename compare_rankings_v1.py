"""Compare the saved Transformer and ranking baselines on identical samples.

python compare_rankings_v1.py
python compare_rankings_v1.py --self-test
python compare_rankings_v1.py --dataset data/example.pt --split all

Default: checkpoint's held-out split, all 500 validation examples.
--split all is ONLY for an independently held-out dataset, not training data.
No training or changes to existing source/data/checkpoints. Each run creates
a unique output folder. Future modifications should use a new version file.

Baselines are explicitly named variants, not claims of exact reproduction of
the Kemeny Transformer paper's implementation or published results:
* Borda: ascending sum of ranks (complete rankings).
* MC4: Dwork et al. (2001), Rank aggregation methods for the Web.
  P[i,j]=1/N if a strict majority prefer j to i, i!=j; diagonal is remainder.
  Use 0.85 damping + uniform teleportation for unique stationary distribution.
* KwikSort: randomized majority-pivot sorting, from Ailon et al. (2008),
  Aggregating inconsistent information: ranking and clustering. 32 restarts,
  retain smallest INPUT disagreement. Not the paper's named KiwiSort package.
* Copeland: pairwise majority wins minus losses.
* Medoid: observed order with minimum summed input Kendall disagreement.
* Kemeny insertion local search: best improving single-item insertion until
  local optimum; starts from Borda, MC4, KwikSort, Copeland and Medoid.
  Heuristic only, NOT exact Kemeny/Mallows MLE or DECoR.
Ties use Borda then item ID, except KwikSort pair ties use item ID.

Metrics use the project's ranking_metrics.py, always versus TRUE CENTER:
normalized Kendall discordance (not correlation), and 1 minus normalized
finite-depth RBO, default p=0.9. No labels enter baseline predictions.
Bootstrap intervals quantify sample variation, not retraining uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(BASE.parent))
    __package__ = BASE.name
from .ranking_metrics import kendall_tau_distance, rbo_distance
from .train import build_model_from_config, split_dataset_indices
from .validation import _checkpoint_model_config

METHODS = ("Transformer", "Borda", "Markov_MC4_damped", "KwikSort_32",
           "Copeland", "Medoid", "Kemeny_insertion_local")
MC_DAMPING = 0.85
MC_TOLERANCE = 1e-12
KWIK_RESTARTS = 32
BOOTSTRAPS = 2000


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else BASE / path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disagreement(order, wins):
    # wins[a,b] = number of observed rankings preferring a to b.
    matrix = wins[np.ix_(order, order)]
    return int(np.tril(matrix, -1).sum())


def mc4(wins, voters, borda):
    n = len(wins)
    transition = (wins.T > voters / 2).astype(float) / n
    np.fill_diagonal(transition, 0)
    transition[np.diag_indices(n)] = 1 - transition.sum(axis=1)
    transition = MC_DAMPING * transition + (1 - MC_DAMPING) / n
    stationary = np.full(n, 1 / n)
    for _ in range(10000):
        updated = stationary @ transition
        if np.abs(updated - stationary).sum() < MC_TOLERANCE:
            stationary = updated
            break
        stationary = updated
    else:
        raise RuntimeError("MC4 stationary iteration did not converge.")
    order = sorted(range(n), key=lambda i: (-round(float(stationary[i]), 12), borda[i], i))
    return order, transition, stationary


def kwiksort(items, wins, rng):
    if len(items) < 2:
        return list(items)
    pivot = int(rng.choice(items))
    left, right = [], []
    for item in items:
        if item == pivot:
            continue
        before = wins[item, pivot] > wins[pivot, item]
        if wins[item, pivot] == wins[pivot, item]:
            before = item < pivot
        (left if before else right).append(item)
    return kwiksort(left, wins, rng) + [pivot] + kwiksort(right, wins, rng)


def insertion_search(initial, wins):
    order = list(initial)
    # Integer cost strictly decreases on each accepted move.
    while True:
        best_delta, move = 0, None
        for i, item in enumerate(order):
            delta = 0
            for j in range(i + 1, len(order)):
                other = order[j]
                delta += int(wins[item, other] - wins[other, item])
                if delta < best_delta:
                    best_delta, move = delta, (i, j)
            delta = 0
            for j in range(i - 1, -1, -1):
                other = order[j]
                delta += int(wins[other, item] - wins[item, other])
                if delta < best_delta:
                    best_delta, move = delta, (i, j)
        if move is None:
            return order
        i, j = move
        item = order.pop(i)
        order.insert(j, item)


def baseline_predictions(positions, seed):
    n, voters = positions.shape
    wins = (positions[:, None, :] < positions[None, :, :]).sum(axis=2).astype(np.int64)
    borda_scores = positions.sum(axis=1)
    borda = sorted(range(n), key=lambda i: (borda_scores[i], i))
    markov, _, _ = mc4(wins, voters, borda_scores)
    copeland_scores = (wins > wins.T).sum(axis=1) - (wins < wins.T).sum(axis=1)
    copeland = sorted(range(n), key=lambda i: (-copeland_scores[i], borda_scores[i], i))
    observed = np.argsort(positions, axis=0).T.tolist()
    medoid = min(observed, key=lambda p: (disagreement(p, wins), p))
    rng = np.random.default_rng(seed)
    candidates = [kwiksort(list(range(n)), wins, rng) for _ in range(KWIK_RESTARTS)]
    kwik = min(candidates, key=lambda p: (disagreement(p, wins), p))
    starts = {tuple(p) for p in (borda, markov, copeland, medoid, kwik)}
    local_candidates = [insertion_search(p, wins) for p in sorted(starts)]
    local = min(local_candidates, key=lambda p: (disagreement(p, wins), p))
    return dict(zip(METHODS[1:], (borda, markov, kwik, copeland, medoid, local)))


def validate_data(payload):
    positions = payload["noisy_positions"].detach().cpu()
    centers = payload["true_rankings"].detach().cpu()
    if positions.ndim != 3 or min(positions.shape) < 1 or centers.shape != positions.shape[:2]:
        raise ValueError("Expected positions [B,N,m] and centers [B,N].")
    b, n, m = positions.shape
    if not torch.equal(positions.sort(dim=1).values, torch.arange(1, n + 1)[None, :, None].expand(b, n, m).to(positions.dtype)):
        raise ValueError("Each observed column must be a complete 1-based rank permutation.")
    if not torch.equal(centers.sort(dim=1).values, torch.arange(n)[None].expand(b, n).to(centers.dtype)):
        raise ValueError("Centers must be permutations of zero-based IDs.")
    if "noisy_rankings" in payload:
        orders = payload["noisy_rankings"].detach().cpu()
        if orders.shape != (b, m, n) or not torch.equal(orders, positions.argsort(dim=1).transpose(1, 2)):
            raise ValueError("noisy_rankings and noisy_positions disagree.")
    return positions.float(), centers.long()


def run(args):
    if not 0 <= args.rbo_p < 1 or args.batch_size < 1 or args.samples < 0:
        raise ValueError("Require 0<=rbo-p<1, batch-size>=1, samples>=0.")
    torch.set_num_threads(1)
    dataset_path, checkpoint_path = resolve(args.dataset), resolve(args.checkpoint)
    payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    positions, centers = validate_data(payload)
    b, n, voters = positions.shape
    saved = checkpoint.get("dataset_split", {})
    if args.split == "checkpoint":
        if not all(key in saved for key in ("dataset_file", "num_samples", "validation_ratio", "split_seed")):
            raise ValueError("Checkpoint lacks split provenance; use --split all only for independent test data.")
        if resolve(saved["dataset_file"]).resolve() != dataset_path.resolve() or saved["num_samples"] != b:
            raise ValueError("Dataset differs from checkpoint split provenance. For independent data use --split all.")
        _, indices = split_dataset_indices(b, float(saved["validation_ratio"]), int(saved["split_seed"]))
    else:
        indices = torch.arange(b)
    if args.samples:
        if args.samples > len(indices):
            raise ValueError("--samples exceeds evaluation split size.")
        permutation = torch.randperm(len(indices), generator=torch.Generator().manual_seed(args.seed))
        indices = indices[permutation[:args.samples]]
    model_config = _checkpoint_model_config(checkpoint["model_config"], n)
    if int(model_config["embedding_dim"]) != voters:
        raise ValueError("Checkpoint embedding dimension must match the observed ranking count.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = build_model_from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    selected_positions = positions.index_select(0, indices)
    selected_centers = centers.index_select(0, indices).tolist()
    predictions = {method: [] for method in METHODS}
    print(f"Evaluating {len(indices)} samples, N={n}, m={voters}, device={device}", flush=True)
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_size):
            batch = selected_positions[start:start + args.batch_size].to(device)
            mask = torch.zeros(batch.shape[:2], dtype=torch.bool, device=device)
            orders, _, _ = model(batch, mask, deterministic=True)
            predictions["Transformer"].extend((orders.cpu() - 1).tolist())
    print("Transformer inference complete.", flush=True)
    for sample, original_index in enumerate(indices.tolist()):
        outputs = baseline_predictions(selected_positions[sample].numpy(), args.seed + original_index)
        for method, order in outputs.items():
            predictions[method].append(order)
        if (sample + 1) % 50 == 0:
            print(f"Baselines: {sample + 1}/{len(indices)}", flush=True)
    scores = {}
    records = []
    for method in METHODS:
        values = []
        for i, (truth, prediction) in enumerate(zip(selected_centers, predictions[method])):
            kd = kendall_tau_distance(truth, prediction)
            rd = rbo_distance(truth, prediction, args.rbo_p)
            values.append((kd, rd))
            records.append({"method": method, "dataset_index": int(indices[i]),
                            "kendall_tau_distance": kd, "rbo_distance": rd,
                            "exact_match": truth == prediction,
                            "true_ranking": truth, "predicted_ranking": prediction})
        scores[method] = np.asarray(values)
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(0, len(indices), size=(BOOTSTRAPS, len(indices)))
    rows = []
    for method in METHODS:
        values = scores[method]
        interval = np.quantile(values[bootstrap_indices].mean(axis=1), [0.025, 0.975], axis=0)
        row = {"method": method, "samples": len(indices)}
        for k, metric in enumerate(("kendall_tau_distance", "rbo_distance")):
            row[f"mean_{metric}"] = float(values[:, k].mean())
            row[f"ci95_low_{metric}"] = float(interval[0, k])
            row[f"ci95_high_{metric}"] = float(interval[1, k])
        row["exact_match_rate"] = sum(a == z for a, z in zip(selected_centers, predictions[method])) / len(indices)
        rows.append(row)
    paired = []
    for method in METHODS[1:]:
        difference = scores["Transformer"] - scores[method]
        interval = np.quantile(difference[bootstrap_indices].mean(axis=1), [0.025, 0.975], axis=0)
        paired.append({"baseline": method, "definition": "Transformer minus baseline; negative favors Transformer",
                       "mean_difference": difference.mean(axis=0).tolist(), "ci95": interval.tolist(),
                       "metric_order": ["kendall_tau_distance", "rbo_distance"]})
    folder = resolve(args.output) / (datetime.now().strftime("v1_%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {
        "dataset": str(dataset_path), "dataset_sha256": file_hash(dataset_path),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_hash(checkpoint_path),
        "script_sha256": file_hash(Path(__file__)), "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "dataset_config": payload.get("config"), "model_config": model_config, "saved_split": saved,
        "split": args.split, "dataset_indices": indices.tolist(), "seed": args.seed,
        "num_items": n, "num_observations": voters, "rbo_p": args.rbo_p,
        "metric_definition": "center recovery; normalized Kendall discordance; 1-normalized finite-depth RBO",
        "mc4_damping": MC_DAMPING, "mc4_tolerance": MC_TOLERANCE,
        "kwiksort_restarts": KWIK_RESTARTS, "bootstrap_replicates": BOOTSTRAPS,
        "device": str(device), "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "notes": [
            "Evaluates the existing checkpoint only; no retraining or baseline tuning on centers.",
            "Baselines are named variants, not a full reproduction of the paper. DECoR and original RL model are not included.",
            "Saved checkpoint has no training-data content hash; split provenance checks path and size only.",
            "Current legacy data generator clips geometric tails; these results do not certify a standard Mallows distribution.",
            "Validation results are exploratory; use an independent test set for final claims after tuning.",
        ],
    }
    (folder / "results.json").write_text(json.dumps({"metadata": metadata, "summary": rows,
        "paired_comparisons": paired}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (folder / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (folder / "per_sample.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    lines = ["# Ranking recovery comparison", "", f"Samples: {len(indices)}; items: {n}; observed rankings: {voters}; RBO p={args.rbo_p}.",
             "", "Both distances are measured against the true center; lower is better.",
             "", "| Method | Kendall distance | RBO distance | Exact recovery |", "|---|---:|---:|---:|"]
    for row in sorted(rows, key=lambda r: r["mean_kendall_tau_distance"]):
        lines.append(f'| {row["method"]} | {row["mean_kendall_tau_distance"]:.6f} | {row["mean_rbo_distance"]:.6f} | {row["exact_match_rate"]:.2%} |')
    lines.extend(["", "Confidence intervals are in summary.csv; paired bootstrap differences are in results.json.", ""])
    lines.extend("- " + note for note in metadata["notes"])
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Results: {folder}", flush=True)
    return folder


def self_test():
    # A nonidentity unanimous center catches orientation and ID-order mistakes.
    center = [3, 1, 4, 0, 2]
    ranks = np.empty(5, dtype=int)
    ranks[center] = np.arange(1, 6)
    position = np.repeat(ranks[:, None], 7, axis=1)
    assert all(p == center for p in baseline_predictions(position, 42).values())
    assert kendall_tau_distance(center, center) == 0
    assert kendall_tau_distance(center, center[::-1]) == 1
    assert rbo_distance(center, center, 0.9) == 0
    assert all(p == [0] for p in baseline_predictions(np.ones((1, 2)), 42).values())
    rng = np.random.default_rng(123)
    for _ in range(8):
        orders = np.array([rng.permutation(5) for _ in range(6)])
        positions = np.argsort(orders, axis=1).T + 1
        wins = (positions[:, None] < positions[None]).sum(axis=2).astype(np.int64)
        predictions = baseline_predictions(positions, 11)
        for p in predictions.values():
            assert sorted(p) == list(range(5))
        _, transition, stationary = mc4(wins, 6, positions.sum(axis=1))
        assert np.allclose(transition.sum(axis=1), 1) and (transition >= 0).all()
        assert np.allclose(stationary @ transition, stationary, atol=1e-11)
        local = predictions["Kemeny_insertion_local"]
        cost = disagreement(local, wins)
        assert cost <= min(disagreement(p, wins) for key, p in predictions.items() if key != "Kemeny_insertion_local")
        for i in range(5):
            for j in range(5):
                candidate = local.copy()
                candidate.insert(j, candidate.pop(i))
                assert disagreement(candidate, wins) >= cost
        for permutation in itertools.permutations(range(5)):
            direct = sum(round(kendall_tau_distance(list(permutation), order.tolist()) * 10) for order in orders)
            assert direct == disagreement(permutation, wins)
    print("PASS: unanimous recovery, metrics, singleton, permutation validity, MC4 stationarity, objective orientation and local optimality.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/ranking_dataset.pt")
    parser.add_argument("--checkpoint", default="checkpoints/mallows_transformer.pt")
    parser.add_argument("--output", default="outputs/comparison")
    parser.add_argument("--split", choices=["checkpoint", "all"], default="checkpoint")
    parser.add_argument("--samples", type=int, default=0, help="0 evaluates the entire selected split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--rbo-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    self_test() if arguments.self_test else run(arguments)
