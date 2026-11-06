# v3 fix: support direct script execution; retain v2 model/data behavior and schema.
# 修正说明：所有指标均比较预测与真实中心；保留旧项目的指标定义。
# Kendall 为归一化逆序对比例而不是相关系数。
# RBO distance = 1 - 有限深度加权重叠 / (1-p**N)，不是外推 RBO。
from __future__ import annotations
import math
import numpy as np


def check_order(order, n):
    value = np.asarray(order)
    if value.shape != (n,) or not np.array_equal(np.sort(value), np.arange(n)):
        raise ValueError("Expected a complete permutation of 0..N-1.")
    return value.astype(np.int64)


def distances(truth, prediction, p=0.9):
    if not 0 <= p < 1:
        raise ValueError("RBO persistence must be in [0,1).")
    n = len(truth)
    if n < 1:
        raise ValueError("Empty ranking.")
    truth, prediction = check_order(truth, n), check_order(prediction, n)
    positions = np.empty(n, dtype=np.int64)
    positions[prediction] = np.arange(n)
    relative = positions[truth]
    i, j = np.triu_indices(n, 1)
    kd = float((relative[i] > relative[j]).mean()) if n > 1 else 0.0
    left, right, total = set(), set(), 0.0
    for depth, (a, b) in enumerate(zip(truth, prediction), 1):
        left.add(int(a))
        right.add(int(b))
        total += p ** (depth - 1) * len(left & right) / depth
    denominator = -math.expm1(n * math.log(p)) if p else 1.0
    rd = min(1.0, max(0.0, 1 - (1 - p) * total / denominator))
    if rd < 1e-12:
        rd = 0.0
    return kd, rd
