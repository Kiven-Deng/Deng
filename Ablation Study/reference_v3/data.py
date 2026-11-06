# v3 fix: support direct script execution; retain v2 model/data behavior and schema.
# 修正说明：
# 1. 先生成互不重复的中心并划分 train/validation/test，再独立生成噪声。
# 2. 默认排除旧数据中的全部中心；同一中心的所有噪声条件只在同一划分。
# 3. 用有限支持上归一化的分阶段几何分布，不再截断普通几何分布的尾部。
# 4. 同一条件生成 max(m) 份观测，各 m 取前缀，便于成对比较。
# 5. 输出分区域真实位移、输入 Kendall/RBO 等诊断；theta 是示例而非现实校准。
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
import math
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, resolve, new_directory, write_json, source_hashes, sha256
from .metrics import distances

PROFILES = ("uniform_low", "uniform_medium", "top_stable", "no_noise")


def theta_profile(n, name):
    if name == "uniform_low":
        return [2.0] * (n - 1)
    if name == "uniform_medium":
        return [1.0] * (n - 1)
    if name == "top_stable":
        return [3.0 if j < math.ceil(0.2 * n) else 1.0 if j < math.ceil(0.6 * n) else 0.35
                for j in range(n - 1)]
    if name == "no_noise":
        return None
    raise ValueError(f"Unknown profile: {name}")


def sample_orders(centers, m, theta, seed, batch_size=128):
    if centers.ndim != 2 or min(centers.shape) < 1 or m < 1 or batch_size < 1:
        raise ValueError("Need nonempty centers [B,N], m>=1 and batch_size>=1.")
    n = centers.shape[1]
    if not torch.equal(centers.sort(1).values, torch.arange(n).expand_as(centers)):
        raise ValueError("Invalid centers.")
    if theta is None:
        return centers[:, None].expand(-1, m, -1).clone()
    if len(theta) != n - 1 or any(not math.isfinite(t) or t < 0 for t in theta):
        raise ValueError("Need N-1 finite nonnegative concentrations.")
    generator = torch.Generator().manual_seed(seed)
    probabilities = [torch.softmax(-t * torch.arange(n - j, dtype=torch.float64), 0)
                     for j, t in enumerate(theta)]
    result = torch.empty((len(centers), m, n), dtype=torch.long)
    for start in range(0, len(centers), batch_size):
        group = centers[start:start + batch_size]
        count = len(group) * m
        available = group[:, None].expand(-1, m, -1).reshape(count, n).clone()
        orders = torch.empty_like(available)
        for j, probability in enumerate(probabilities):
            choice = torch.multinomial(probability, count, replacement=True, generator=generator)
            orders[:, j] = available.gather(1, choice[:, None]).squeeze(1)
            indices = torch.arange(available.shape[1] - 1).expand(count, -1)
            indices = indices + (indices >= choice[:, None]).long()
            available = available.gather(1, indices)
        orders[:, -1] = available[:, 0]
        result[start:start + len(group)] = orders.reshape(len(group), m, n)
    return result


def orders_to_positions(orders):
    return torch.empty_like(orders).scatter_(
        2, orders, torch.arange(1, orders.shape[-1] + 1).expand_as(orders)
    ).transpose(1, 2).contiguous().float()


def build_bundle(n=50, sizes=(4000, 500, 500), m_values=(2, 3, 5, 10),
                 profiles=("uniform_low", "uniform_medium", "top_stable"),
                 seed=20261106, excluded=()):
    if n < 2 or len(sizes) != 3 or min(sizes) < 1:
        raise ValueError("N>=2 and three positive split sizes are required.")
    if not m_values or min(m_values) < 1 or len(set(m_values)) != len(m_values):
        raise ValueError("Provide distinct positive observation counts.")
    if not profiles or len(set(profiles)) != len(profiles) or not set(profiles) <= set(PROFILES):
        raise ValueError("Invalid or repeated profiles.")
    occupied = {tuple(row) for row in excluded if len(row) == n}
    if sum(sizes) + len(occupied) > math.factorial(n):
        raise ValueError("Not enough distinct permutations for the requested split sizes.")
    generator = torch.Generator().manual_seed(seed)
    rows = []
    while len(rows) < sum(sizes):
        row = tuple(torch.randperm(n, generator=generator).tolist())
        if row not in occupied:
            occupied.add(row)
            rows.append(row)
    centers = torch.tensor(rows, dtype=torch.long)
    splits, offset, noise_seeds = {}, 0, {}
    for s, (name, count) in enumerate(zip(("train", "validation", "test"), sizes)):
        truth = centers[offset:offset + count].clone()
        offset += count
        conditions = {}
        for k, profile in enumerate(profiles):
            noise_seed = seed + 100003 + 1009 * s + 10007 * k
            theta = theta_profile(n, profile)
            orders = sample_orders(truth, max(m_values), theta, noise_seed)
            conditions[profile] = {"noisy_rankings": orders,
                                   "noisy_positions": orders_to_positions(orders),
                                   "theta_vector": theta}
            noise_seeds[f"{name}/{profile}"] = noise_seed
        splits[name] = {"true_rankings": truth, "conditions": conditions}
    return {"format": "ranking_recovery_v2", "splits": splits,
            "config": {"num_items": n, "split_sizes": dict(zip(splits, sizes)),
                       "m_values": list(m_values), "profiles": list(profiles),
                       "seed": seed, "noise_seeds": noise_seeds,
                       "theta_semantics": "output selection stages, not final item displacement",
                       "source_hashes": source_hashes()}}


def validate_bundle(bundle):
    if bundle.get("format") != "ranking_recovery_v2":
        raise ValueError("Expected a v2 dataset; regenerate with ranking_recovery_v2.data.")
    n, seen = int(bundle["config"]["num_items"]), set()
    maximum = max(bundle["config"]["m_values"])
    for name in ("train", "validation", "test"):
        split = bundle["splits"][name]
        truth = split["true_rankings"]
        if truth.ndim != 2 or truth.shape[1] != n or len(truth) < 1:
            raise ValueError("Invalid split shape.")
        if not torch.equal(truth.sort(1).values, torch.arange(n).expand_as(truth)):
            raise ValueError("Invalid center permutation.")
        rows = {tuple(row) for row in truth.tolist()}
        if len(rows) != len(truth) or rows & seen:
            raise ValueError("Repeated center within/across splits.")
        seen.update(rows)
        if set(split["conditions"]) != set(bundle["config"]["profiles"]):
            raise ValueError("Profile mismatch.")
        for condition in split["conditions"].values():
            positions, orders = condition["noisy_positions"], condition["noisy_rankings"]
            if positions.shape != (len(truth), n, maximum) or orders.shape != (len(truth), maximum, n):
                raise ValueError("Invalid observation shape.")
            expected = torch.arange(n).expand_as(orders)
            if not torch.equal(orders.sort(-1).values, expected):
                raise ValueError("Invalid observed permutation.")
            if not torch.equal(positions, orders_to_positions(orders)):
                raise ValueError("Orders/positions mismatch.")


def diagnostics(bundle, limit=100, p=0.9):
    records = []
    n = bundle["config"]["num_items"]
    boundaries = [0, math.ceil(.2 * n), math.ceil(.6 * n), n]
    for name, split in bundle["splits"].items():
        truth = split["true_rankings"][:limit]
        truepos = torch.empty_like(truth).scatter_(1, truth, torch.arange(n).expand_as(truth))
        for profile, condition in split["conditions"].items():
            for m in bundle["config"]["m_values"]:
                orders = condition["noisy_rankings"][:len(truth), :m]
                pos = condition["noisy_positions"][:len(truth), :, :m]
                displacement = (pos - 1 - truepos[:, :, None]).abs().mean(-1).gather(1, truth)
                values = [distances(t.tolist(), order.tolist(), p)
                          for t, observations in zip(truth, orders) for order in observations]
                means = np.mean(values, axis=0)
                regions = [float(displacement[:, a:b].mean()) if b > a else None
                           for a, b in zip(boundaries[:-1], boundaries[1:])]
                records.append({"split": name, "profile": profile, "m": m,
                                "centers_checked": len(truth), "rbo_p": p,
                                "mean_input_kendall": float(means[0]), "mean_input_rbo": float(means[1]),
                                "mean_absolute_displacement_top_middle_tail": regions})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=50)
    parser.add_argument("--sizes", type=int, nargs=3, default=[4000, 500, 500])
    parser.add_argument("--m-values", type=int, nargs="+", default=[2, 3, 5, 10])
    parser.add_argument("--profiles", nargs="+", choices=PROFILES,
                        default=["uniform_low", "uniform_medium", "top_stable"])
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--output", default="data/ranking_recovery_v2")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Old .pt center datasets to exclude; default discovers existing legacy datasets.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = ([resolve(p) for p in args.exclude] if args.exclude is not None else
             [p for p in [ROOT / "data/ranking_dataset.pt", ROOT / "data/true_rankings.pt",
                          *sorted((ROOT / "data/generalized_mallows").glob("*.pt"))] if p.exists()])
    excluded, provenance = [], []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        truth = payload["true_rankings"] if isinstance(payload, dict) else payload
        excluded.extend(truth.tolist())
        provenance.append({"path": str(path), "sha256": sha256(path)})
    bundle = build_bundle(args.items, args.sizes, args.m_values, args.profiles, args.seed, excluded)
    bundle["config"]["excluded_datasets"] = provenance
    validate_bundle(bundle)
    folder = new_directory(args.output, "dataset")
    with (folder / "dataset.pt").open("xb") as stream:
        torch.save(bundle, stream)
    write_json(folder / "manifest.json", bundle["config"])
    write_json(folder / "diagnostics.json", diagnostics(bundle))
    print(f"Dataset: {folder / 'dataset.pt'}", flush=True)
    print("Disjoint centers verified; diagnostics and manifest saved.", flush=True)


if __name__ == "__main__":
    main()
