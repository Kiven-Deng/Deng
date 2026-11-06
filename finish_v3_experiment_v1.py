# 修正/用途说明：
# 1. 仅在训练写出 best.json 后评估，不抢占或重复启动正在运行的训练。
# 2. 独立锁与状态文件避免自动任务重复评估；保留全部现有代码和权重。
# 3. 使用完整 test 和全部预定噪声/m 条件，不根据测试结果调参。
# 4. 对旧权重追加同一测试中心、相同观测的 m=10 对照，避免跨数据直接比较。
# 5. 报告全部基线及限制；比较整套训练结果，不据此单独归因于某个结构改动。
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ranking_recovery_v3.evaluate import evaluate
from ranking_recovery_v3.common import sha256
from ranking_recovery_v3.metrics import distances


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def compare_legacy(checkpoint_path, data_path, result, output):
    sys.path.insert(0, str(ROOT.parent))
    from Deng.train import build_model_from_config
    from Deng.validation import _checkpoint_model_config
    legacy_path = ROOT / "checkpoints/mallows_transformer.pt"
    legacy = torch.load(legacy_path, map_location="cpu", weights_only=True)
    bundle = torch.load(data_path, map_location="cpu", weights_only=True)
    n = bundle["config"]["num_items"]
    config = _checkpoint_model_config(legacy["model_config"], n)
    m = int(config["embedding_dim"])
    if m not in result["m_values"]:
        return {"skipped": f"Legacy model requires m={m}, absent from this test protocol."}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config).to(device).eval()
    model.load_state_dict(legacy["model_state_dict"], strict=True)
    ids = torch.tensor(result["center_indices"], dtype=torch.long)
    section = bundle["splits"]["test"]
    truth = section["true_rankings"][ids].tolist()
    rows = []
    with (output / "legacy_same_test_per_sample.jsonl").open("x", encoding="utf-8") as stream:
        for profile, condition in section["conditions"].items():
            x = condition["noisy_positions"][ids, :, :m]
            predictions = []
            with torch.inference_mode():
                for start in range(0, len(ids), 32):
                    batch = x[start:start + 32].to(device)
                    mask = torch.zeros(batch.shape[:2], dtype=torch.bool, device=device)
                    order, _, _ = model(batch, mask, deterministic=True)
                    predictions.extend((order.cpu() - 1).tolist())
            values = np.array([distances(t, p, result["rbo_p"]) for t, p in zip(truth, predictions)])
            current = next(r for r in result["summary"] if r["profile"] == profile and r["m"] == m and r["method"] == "Transformer")
            rows.append({"profile": profile, "m": m, "samples": len(ids),
                         "old_kendall": float(values[:, 0].mean()), "old_rbo": float(values[:, 1].mean()),
                         "new_kendall": current["kendall"], "new_rbo": current["rbo"]})
            for j, (target, prediction) in enumerate(zip(truth, predictions)):
                stream.write(json.dumps({"center_index": int(ids[j]), "profile": profile, "m": m,
                             "true_ranking": target, "prediction": prediction,
                             "kendall": float(values[j, 0]), "rbo": float(values[j, 1])}) + "\n")
            print(f"Legacy same-test comparison complete: {profile}, m={m}", flush=True)
    report = {"legacy_checkpoint": str(legacy_path), "legacy_sha256": sha256(legacy_path),
              "new_checkpoint": str(checkpoint_path), "dataset": str(data_path), "rows": rows,
              "limitation": "Same test inputs, but training distributions and architectures differ. This measures whole-system improvement, not an isolated causal ablation."}
    (output / "legacy_same_test.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def report_text(result, legacy, history, checkpoint):
    rows = result["summary"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    lines = ["# Completed v3 experiment", "",
             f"Checkpoint: {checkpoint}", f"Test centers: {len(result['center_indices'])}",
             f"RBO p={result['rbo_p']}. Lower distances are better.",
             "", "## Macro averages over the prespecified conditions", "",
             "| Method | Kendall | RBO |", "|---|---:|---:|"]
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        lines.append(f"| {method} | {np.mean([r['kendall'] for r in subset]):.6f} | {np.mean([r['rbo'] for r in subset]):.6f} |")
    lines.extend(["", "## All conditions", "",
                  "| Profile | m | Transformer K | Borda K | Lowest baseline K | Transformer RBO | Borda RBO | Lowest baseline RBO |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    current = [r for r in rows if r["method"] == "Transformer"]
    for r in current:
        baselines = [b for b in rows if b["profile"] == r["profile"] and b["m"] == r["m"] and b["method"] != "Transformer"]
        borda = next(b for b in baselines if b["method"] == "Borda")
        lines.append(f"| {r['profile']} | {r['m']} | {r['kendall']:.6f} | {borda['kendall']:.6f} | {min(b['kendall'] for b in baselines):.6f} | {r['rbo']:.6f} | {borda['rbo']:.6f} | {min(b['rbo'] for b in baselines):.6f} |")
    if "rows" in legacy:
        lines.extend(["", "## Old and new checkpoints on identical test observations", "",
                      "| Profile | m | Old Kendall | New Kendall | Old RBO | New RBO |",
                      "|---|---:|---:|---:|---:|---:|"])
        for r in legacy["rows"]:
            lines.append(f"| {r['profile']} | {r['m']} | {r['old_kendall']:.6f} | {r['new_kendall']:.6f} | {r['old_rbo']:.6f} | {r['new_rbo']:.6f} |")
    lines.extend(["", "## Interpretation limits", "",
                  "- Best checkpoint was selected by validation, not test.",
                  "- Same-test old/new comparison changes both training data and architecture; it cannot isolate LayerNorm or any single mechanism.",
                  "- One training seed does not establish robust superiority. Paired bootstrap intervals in results.json only quantify sampled-center uncertainty.",
                  "- No hyperparameter search or retraining was driven by this test result.",
                  "- All prescribed conditions and baselines are retained, including conditions where the model loses.",
                  f"- Training completed {len(history)} recorded epochs.",
                  "- Earlier legacy-table results used different data and are not directly comparable to this experiment."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    train_folders = list((root / "checkpoints").glob("train_*"))
    if len(train_folders) != 1:
        raise ValueError("Expected exactly one training run.")
    training = train_folders[0]
    best_path = training / "best.json"
    if not best_path.exists():
        print("WAITING: training has not written best.json.", flush=True)
        return
    state_path = root / "auto_completion_v1.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["status"] == "complete":
            print(json.dumps(state, ensure_ascii=False), flush=True)
            return
        if state["status"] == "error":
            raise RuntimeError(f"Previous completion attempt failed. Inspect {state_path}; do not silently retry.")
    lock = root / "auto_completion_v1.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started": datetime.now().isoformat()}, stream)
    except FileExistsError:
        print("BUSY: completion worker lock already exists; no duplicate launch.", flush=True)
        return
    try:
        chosen = read_json(best_path)
        checkpoint = Path(chosen["checkpoint"])
        manifest = read_json(training / "manifest.json")
        data_path = Path(manifest["dataset_path"])
        for name, expected in manifest["source_hashes"].items():
            if sha256(ROOT / "ranking_recovery_v3" / name) != expected:
                raise RuntimeError(f"Source changed after training began: {name}")
        save_state(state_path, {"status": "running", "pid": os.getpid(), "checkpoint": str(checkpoint)})
        # Reuse an already-completed fourth step if the user ran it manually.
        result_paths = list((root / "comparison").glob("*/results.json"))
        result_paths += list((root / "comparison_auto").glob("*/results.json"))
        matching = []
        for path in result_paths:
            value = read_json(path)
            if (value.get("checkpoint_sha256") == sha256(checkpoint) and value.get("split") == "test"
                    and value.get("dataset_sha256") == manifest["dataset_sha256"] and len(value["center_indices"]) == 500
                    and set(value["m_values"]) == set(manifest["m_values"])):
                matching.append(path)
        if matching:
            output = matching[0].parent
        else:
            output = evaluate(argparse.Namespace(checkpoint=str(checkpoint), dataset=str(data_path),
                              output=str(root / "comparison_auto"), split="test", m_values=None,
                              samples=0, batch_size=32, rbo_p=.9, seed=2026, device="auto"))
        result = read_json(output / "results.json")
        legacy = compare_legacy(checkpoint, data_path, result, output)
        history = [json.loads(line) for line in (training / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        report = output / "analysis_report.md"
        with report.open("x", encoding="utf-8") as stream:
            stream.write(report_text(result, legacy, history, checkpoint))
        save_state(state_path, {"status": "complete", "checkpoint": str(checkpoint),
                               "results": str(output / "results.json"), "report": str(report),
                               "legacy_comparison": str(output / "legacy_same_test.json"),
                               "completed": datetime.now().isoformat()})
        print(f"COMPLETE: {report}", flush=True)
    except Exception:
        save_state(state_path, {"status": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
