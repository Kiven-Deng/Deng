# v3 fix: support direct script execution; retain v2 model/data behavior and schema.
# 修正说明：先检查分布、划分和结构性质，再训练正式模型。
# 自检不改动旧文件，不生成正式训练数据。微型拟合只用于检查可学习性。
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
import itertools
import math
import tempfile
from pathlib import Path
import numpy as np
import torch
from .data import build_bundle, validate_bundle, sample_orders, orders_to_positions
from .model import ModelConfig, RecoveryTransformer
from .metrics import distances
from .baselines import baseline_predictions


def run_checks(overfit=True):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    bundle = build_bundle(6, (24, 8, 8), (1, 2, 4), ("top_stable", "no_noise"), 123)
    validate_bundle(bundle)
    all_rows = [tuple(row) for split in bundle["splits"].values() for row in split["true_rankings"].tolist()]
    assert len(set(all_rows)) == 40
    section = bundle["splits"]["train"]
    center = section["true_rankings"][:8]
    x = section["conditions"]["top_stable"]["noisy_positions"][:8]
    for profile, concentration in (("uniform", [0.] * 3), ("constant", [.7] * 3), ("vector", [2., .6, .1])):
        permutations = list(itertools.permutations(range(4)))
        weights = []
        for permutation in permutations:
            remaining, energy = list(range(4)), 0.
            for j, item in enumerate(permutation[:-1]):
                energy += concentration[j] * remaining.index(item)
                remaining.remove(item)
            weights.append(math.exp(-energy))
        expected = np.array(weights) / sum(weights)
        draws = sample_orders(torch.arange(4)[None], 40000, concentration, 77)[0].numpy()
        observed = np.array([(draws == permutation).all(1).mean() for permutation in permutations])
        tolerance = 6 * np.sqrt(expected * (1 - expected) / len(draws)) + 2 / len(draws)
        assert (abs(expected - observed) <= tolerance).all(), profile
    for rarope, anlibi in ((False, False), (True, False), (False, True), (True, True)):
        model = RecoveryTransformer(ModelConfig(hidden=32, heads=4, encoder_layers=1,
                                               decoder_layers=1, rarope=rarope, anlibi=anlibi))
        model.eval()
        logits = model(x, center)
        assert torch.allclose(logits, model(x[:, :, [3, 0, 2, 1]], center), atol=3e-6, rtol=1e-5)
        assert torch.allclose(logits[:1], model(x[:1], center[:1]), atol=3e-6, rtol=1e-5)
        model.train()
        assert torch.allclose(logits, model(x, center), atol=3e-6, rtol=1e-5)
        model.eval()
        # Object relabeling changes rows and item labels together.
        permutation = torch.tensor([2, 5, 0, 4, 1, 3])
        inverse = permutation.argsort()
        remapped = model(x[:, permutation], inverse[center])
        assert torch.allclose(remapped, logits[:, :, permutation], atol=3e-6, rtol=1e-5)
        # Different future labels cannot alter earlier predictions.
        changed = center.clone()
        changed[:, 3:] = changed[:, 3:].flip(1)
        assert torch.allclose(logits[:, :4], model(x, changed)[:, :4], atol=3e-6, rtol=1e-5)
        memory, ranks, summary = model.encode(x)
        for t in range(6):
            serial = model.prefix_logits(memory, ranks, summary, center[:, :t])[:, -1]
            assert torch.allclose(serial, logits[:, t], atol=3e-6, rtol=1e-5)
        for m in (1, 2, 4):
            prediction = model.predict(x[:, :, :m])
            assert torch.equal(prediction.sort(1).values, torch.arange(6).expand(8, -1))
        model.zero_grad(set_to_none=True)
        model.loss(x, center).backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    observation, summary = model.features(torch.tensor([[[1., 2., 1., 1.], [21., 22., 21., 21.]]]))
    assert not torch.allclose(observation[:, 0], observation[:, 1])
    assert not torch.allclose(summary[:, 0], summary[:, 1])
    assert distances([0, 1, 2], [0, 1, 2]) == (0., 0.)
    assert distances([0, 1, 2], [2, 1, 0])[0] == 1
    unanimous = section["conditions"]["no_noise"]["noisy_positions"][0].numpy()
    assert all(order == section["true_rankings"][0].tolist() for order in baseline_predictions(unanimous, 42).values())
    with tempfile.TemporaryDirectory(prefix="recovery_v2_check_") as directory:
        path = Path(directory) / "weights.pt"
        torch.save(model.state_dict(), path)
        copy = RecoveryTransformer(model.config)
        copy.load_state_dict(torch.load(path, weights_only=True))
        assert torch.equal(model.predict(x), copy.eval().predict(x))
    print("PASS: exact small-N law, disjoint centers, column invariance, object equivariance, batch independence, train/eval agreement, causal decoder, all ablations, gradients, checkpoint roundtrip.", flush=True)
    if overfit:
        torch.manual_seed(42)
        model = RecoveryTransformer(ModelConfig(hidden=32, heads=4, encoder_layers=1, decoder_layers=1))
        target = section["true_rankings"]
        inputs = section["conditions"]["no_noise"]["noisy_positions"][:, :, :2]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        passed = False
        for step in range(1, 401):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(inputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step % 25 == 0:
                model.eval()
                predicted = model.predict(inputs)
                exact = float((predicted == target).all(1).float().mean())
                print(f"Micro-fit step={step} CE={float(loss.detach()):.5f} exact={exact:.3f}", flush=True)
                if exact == 1.0 and float(loss.detach()) < .1:
                    passed = True
                    break
        assert passed, "No-noise micro-fit failed; do not start a formal experiment."
        print("PASS: no-noise micro-fit and free decoding recovered all 24 centers.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-overfit", action="store_true")
    args = parser.parse_args()
    run_checks(not args.skip_overfit)
