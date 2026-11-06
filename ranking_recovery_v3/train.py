# v3 fix: support direct script execution; retain v2 model/data behavior and schema.
# 修正说明：
# 1. 独立训练新模型，不加载或覆盖旧 BatchNorm 权重。
# 2. 训练时在不同噪声条件和 m 间采样；隐藏维度始终固定。
# 3. 每轮 eval() 自由解码验证，按宏平均 Kendall 选权重，RBO 仅作次级平局指标。
# 4. 学习率调度、提前停止和最佳权重选择只使用 validation，不使用 test。
# 5. 写入新目录，记录数据/代码指纹、配置、每轮指标和具体 checkpoint。
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
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import resolve, new_directory, write_json, sha256, source_hashes
from .data import validate_bundle
from .metrics import distances
from .model import ModelConfig, RecoveryTransformer


def condition_metrics(model, bundle, split, m_values, device, batch_size=32, limit=0):
    model.eval()
    section = bundle["splits"][split]
    truth = section["true_rankings"]
    if limit:
        truth = truth[:limit]
    rows = []
    with torch.inference_mode():
        for profile, condition in section["conditions"].items():
            for m in m_values:
                scores, ce_sum = [], 0.0
                for start in range(0, len(truth), batch_size):
                    target = truth[start:start + batch_size].to(device)
                    x = condition["noisy_positions"][start:start + len(target), :, :m].to(device)
                    ce_sum += float(model.loss(x, target)) * len(target)
                    predicted = model.predict(x).cpu().tolist()
                    scores.extend(distances(a, b) for a, b in zip(target.cpu().tolist(), predicted))
                values = np.asarray(scores)
                rows.append({"profile": profile, "m": m, "samples": len(truth),
                             "kendall": float(values[:, 0].mean()),
                             "rbo": float(values[:, 1].mean()),
                             "teacher_forcing_ce": ce_sum / len(truth)})
    return rows


def train(args):
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0 or args.max_batches < 0 or args.val_limit < 0:
        raise ValueError("Invalid training sizes, limits or learning rate.")
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    path = resolve(args.dataset)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    validate_bundle(bundle)
    m_values = args.m_values or bundle["config"]["m_values"]
    if not set(m_values) <= set(bundle["config"]["m_values"]):
        raise ValueError("m-values must be included in the dataset protocol.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    config = ModelConfig(hidden=args.hidden, heads=args.heads, encoder_layers=args.encoder_layers,
                         decoder_layers=args.decoder_layers, rarope=args.rarope, anlibi=args.anlibi)
    model = RecoveryTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=3, min_lr=1e-6)
    folder = new_directory(args.output, "train")
    dataset_hash = sha256(path)
    metadata = {"dataset_path": str(path), "dataset_sha256": dataset_hash,
                "model_config": config.to_dict(), "training_args": vars(args),
                "m_values": m_values, "selection": "macro validation Kendall; RBO breaks exact ties",
                "validation_center_indices": list(range(min(args.val_limit or 10**12,
                                                            len(bundle["splits"]["validation"]["true_rankings"])))),
                "metrics": "normalized Kendall and normalized finite-depth RBO distance, p=0.9",
                "source_hashes": source_hashes(), "device": str(device),
                "torch_version": str(torch.__version__), "test_used_for_selection": False}
    write_json(folder / "manifest.json", metadata)
    section = bundle["splits"]["train"]
    truth = section["true_rankings"]
    profiles = list(section["conditions"])
    count = len(truth)
    generator = torch.Generator().manual_seed(args.seed + 1)
    rng = np.random.default_rng(args.seed + 2)
    best, stale, best_path, history = (math.inf, math.inf), 0, None, []
    print(f"Run: {folder}\nDevice: {device}; parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Each center/profile pair occurs once per full epoch; m is drawn per batch.
        permutation = torch.randperm(count * len(profiles), generator=generator)
        loss_sum, samples, steps = 0.0, 0, 0
        for start in range(0, len(permutation), args.batch_size):
            flat = permutation[start:start + args.batch_size]
            ids, condition_ids = flat % count, flat // count
            m = int(rng.choice(m_values))
            x = torch.stack([section["conditions"][profiles[int(k)]]["noisy_positions"][int(i), :, :m]
                             for i, k in zip(ids, condition_ids)]).to(device)
            target = truth[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite loss; training stopped without touching old runs.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            samples += len(ids)
            steps += 1
            if steps % 50 == 0:
                print(f"epoch={epoch} batch={steps} train_ce={loss_sum / samples:.5f}", flush=True)
            if args.max_batches and steps >= args.max_batches:
                break
        rows = condition_metrics(model, bundle, "validation", m_values, device,
                                 args.batch_size, args.val_limit)
        key = (float(np.mean([r["kendall"] for r in rows])),
               float(np.mean([r["rbo"] for r in rows])))
        scheduler.step(key[0])
        improved = key < best
        if improved:
            best, stale = key, 0
            best_path = folder / f"best_epoch_{epoch:03d}.pt"
            with best_path.open("xb") as stream:
                torch.save({"format": "ranking_recovery_v2", "model_state_dict": model.state_dict(),
                            "model_config": config.to_dict(), "epoch": epoch,
                            "validation_key": list(key), "metadata": metadata,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()}, stream)
        else:
            stale += 1
        record = {"epoch": epoch, "train_ce": loss_sum / samples, "validation": rows,
                  "macro_kendall": key[0], "macro_rbo": key[1], "improved": improved,
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "elapsed_seconds": time.perf_counter() - start_time}
        history.append(record)
        with (folder / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"epoch={epoch} CE={record['train_ce']:.5f} val_K={key[0]:.6f} val_RBO={key[1]:.6f} best={improved}", flush=True)
        if stale >= args.patience:
            print("Early stopping on validation.", flush=True)
            break
    write_json(folder / "best.json", {"checkpoint": str(best_path), "validation_key": list(best),
                                    "epochs_completed": len(history), "test_evaluated": False})
    print(f"Best checkpoint: {best_path}", flush=True)
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="checkpoints/ranking_recovery_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-limit", type=int, default=0, help="0=all validation centers; positive for smoke checks")
    parser.add_argument("--max-batches", type=int, default=0, help="0=full epoch")
    parser.add_argument("--m-values", type=int, nargs="+")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--rarope", action="store_true")
    parser.add_argument("--anlibi", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
