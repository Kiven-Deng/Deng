# v3 fix: support direct script execution; retain v2 model/data behavior and schema.
# 修正说明：
# 1. 默认只评估预先隔离的 test；不会调用旧训练/验证划分。
# 2. 同一模型在不同 m 和噪声配置上评估，各方法共享相同中心与观测前缀。
# 3. 核对 checkpoint 与数据指纹，防止错用数据或旧权重。
# 4. 输出逐样本结果、分条件汇总、置信区间和按中心聚合的成对差值。
# 5. 基线为明确命名的算法变体，不冒充完整复现原论文。
from __future__ import annotations

# Direct-script compatibility: establish the package before relative imports.
# Works with IDE Run File, absolute paths and python -m alike.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _package_directory = _Path(__file__).resolve().parent
    _sys.path.insert(0, str(_package_directory.parent))
    __package__ = _package_directory.name

import argparse
import csv
import json
import numpy as np
import torch
from .common import resolve, sha256, write_json, new_directory, source_hashes
from .data import validate_bundle
from .model import RecoveryTransformer, ModelConfig
from .metrics import distances
from .baselines import METHODS, baseline_predictions


def evaluate(args):
    if args.samples < 0 or args.batch_size < 1 or not 0 <= args.rbo_p < 1:
        raise ValueError("Invalid sample count, batch size or RBO p.")
    torch.set_num_threads(1)
    checkpoint_path = resolve(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "ranking_recovery_v2":
        raise ValueError("Use a newly trained v2 checkpoint, not the old architecture.")
    path = resolve(args.dataset or checkpoint["metadata"]["dataset_path"])
    if sha256(path) != checkpoint["metadata"]["dataset_sha256"]:
        raise ValueError("Dataset fingerprint differs from the checkpoint's split protocol.")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    validate_bundle(bundle)
    m_values = args.m_values or bundle["config"]["m_values"]
    if not set(m_values) <= set(bundle["config"]["m_values"]):
        raise ValueError("m-values must be declared in the dataset protocol.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = RecoveryTransformer(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    section = bundle["splits"][args.split]
    truth = section["true_rankings"]
    if args.samples > len(truth):
        raise ValueError("Requested too many centers.")
    count = args.samples or len(truth)
    ids = torch.randperm(len(truth), generator=torch.Generator().manual_seed(args.seed))[:count]
    targets = truth[ids].tolist()
    folder = new_directory(args.output, "comparison")
    rng = np.random.default_rng(args.seed)
    bootstrap = rng.integers(0, count, size=(2000, count))
    summary, condition_scores = [], []
    with (folder / "per_sample.jsonl").open("x", encoding="utf-8") as details:
        for profile, condition in section["conditions"].items():
            for m in m_values:
                x = condition["noisy_positions"][ids, :, :m]
                predictions = {method: [] for method in METHODS}
                with torch.inference_mode():
                    for start in range(0, count, args.batch_size):
                        predictions["Transformer"].extend(
                            model.predict(x[start:start + args.batch_size].to(device)).cpu().tolist())
                for j in range(count):
                    baseline = baseline_predictions(x[j].numpy(), args.seed + int(ids[j]))
                    for method, order in baseline.items():
                        predictions[method].append(order)
                all_values = []
                for method in METHODS:
                    values = np.array([distances(t, o, args.rbo_p) for t, o in zip(targets, predictions[method])])
                    all_values.append(values)
                    ci = np.quantile(values[bootstrap].mean(1), [.025, .975], axis=0)
                    row = {"split": args.split, "profile": profile, "m": m, "method": method,
                           "samples": count, "kendall": float(values[:, 0].mean()),
                           "rbo": float(values[:, 1].mean()), "kendall_ci_low": float(ci[0, 0]),
                           "kendall_ci_high": float(ci[1, 0]), "rbo_ci_low": float(ci[0, 1]),
                           "rbo_ci_high": float(ci[1, 1]),
                           "exact_recovery": sum(t == o for t, o in zip(targets, predictions[method])) / count}
                    summary.append(row)
                    for j, (target, predicted) in enumerate(zip(targets, predictions[method])):
                        details.write(json.dumps({"split": args.split, "center_index": int(ids[j]),
                            "profile": profile, "m": m, "method": method, "true_ranking": target,
                            "prediction": predicted, "kendall": float(values[j, 0]), "rbo": float(values[j, 1])}) + "\n")
                condition_scores.append(np.stack(all_values))
                print(f"{profile} m={m}: Transformer K={summary[-len(METHODS)]['kendall']:.6f}, RBO={summary[-len(METHODS)]['rbo']:.6f}", flush=True)
    # Average conditions WITHIN each center before bootstrapping. A center
    # repeated under several profiles/m values is not an independent sample.
    center_means = np.stack(condition_scores).mean(0)
    paired = []
    for j, method in enumerate(METHODS[1:], 1):
        difference = center_means[0] - center_means[j]
        paired.append({"baseline": method, "metric_order": ["kendall", "rbo"],
                       "mean_transformer_minus_baseline": difference.mean(0).tolist(),
                       "ci95": np.quantile(difference[bootstrap].mean(1), [.025, .975], axis=0).tolist()})
    with (folder / "summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(folder / "results.json", {
        "dataset": str(path), "dataset_sha256": sha256(path), "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "split": args.split,
        "center_indices": ids.tolist(), "rbo_p": args.rbo_p, "seed": args.seed,
        "m_values": m_values, "summary": summary, "macro_paired_by_center": paired,
        "source_hashes": source_hashes(),
        "notes": ["No tuning is performed by this script.",
                  "Do not use test results to select architecture/hyperparameters.",
                  "Metrics follow the original normalized finite-depth RBO convention.",
                  "Baseline variants: MC4 damping=0.85, KwikSort restarts=32, heuristic insertion search.",
                  "Bootstrap intervals cover sampled centers, not retraining randomness."]})
    print(f"Results: {folder}", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--output", default="outputs/ranking_recovery_v2")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--m-values", nargs="+", type=int)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rbo-p", type=float, default=.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
