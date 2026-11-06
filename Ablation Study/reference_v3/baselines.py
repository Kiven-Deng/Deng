# v3 fix: support direct script execution; retain v2 model/data behavior and schema.
# Revision notes: independent snapshot of the already tested v1 baselines.
# No original model/train imports or global RNG side effects.
# Keeps the exact baseline variants and tie rules used in the previous comparison.
# MC4 uses damping 0.85; KwikSort uses 32 input-objective restarts.
# Kemeny insertion search is a heuristic, not DECoR or exact MLE.
import numpy as np
METHODS = ("Transformer", "Borda", "Markov_MC4_damped", "KwikSort_32", "Copeland", "Medoid", "Kemeny_insertion_local")
MC_DAMPING = 0.85
MC_TOLERANCE = 1e-12
KWIK_RESTARTS = 32

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
