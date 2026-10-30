"""Distance metrics for complete item-order rankings."""

from __future__ import annotations

from typing import Sequence


def _validate_ranking_pair(
    true_ranking: Sequence[int],
    predicted_ranking: Sequence[int],
) -> None:
    if len(true_ranking) != len(predicted_ranking):
        raise ValueError("Rankings must contain the same number of items.")
    if not true_ranking:
        raise ValueError("Rankings must not be empty.")
    if len(set(true_ranking)) != len(true_ranking):
        raise ValueError("The true ranking contains duplicate item IDs.")
    if len(set(predicted_ranking)) != len(predicted_ranking):
        raise ValueError("The predicted ranking contains duplicate item IDs.")
    if set(true_ranking) != set(predicted_ranking):
        raise ValueError("Rankings must contain the same item IDs.")


def kendall_tau_distance(
    true_ranking: Sequence[int],
    predicted_ranking: Sequence[int],
) -> float:
    """Return normalized Kendall tau distance in ``[0, 1]``.

    The value is the fraction of item pairs ordered differently by the two
    rankings. Identical full rankings score 0 and reversed rankings score 1.
    """
    _validate_ranking_pair(true_ranking, predicted_ranking)
    num_items = len(true_ranking)
    if num_items < 2:
        return 0.0

    predicted_positions = {
        item_id: position
        for position, item_id in enumerate(predicted_ranking)
    }
    discordant_pairs = 0
    for left in range(num_items):
        left_item = true_ranking[left]
        for right in range(left + 1, num_items):
            right_item = true_ranking[right]
            if (
                predicted_positions[left_item]
                > predicted_positions[right_item]
            ):
                discordant_pairs += 1

    total_pairs = num_items * (num_items - 1) // 2
    return discordant_pairs / total_pairs


def rbo_similarity(
    true_ranking: Sequence[int],
    predicted_ranking: Sequence[int],
    persistence: float,
) -> float:
    """Return normalized, finite-depth Rank-Biased Overlap.

    The weighted overlap is:

    ``RBO = (1-p) * sum_d p^(d-1) * overlap_at_depth_d / d``

    For a finite complete ranking, the raw value is normalized by the
    identical-ranking maximum ``1 - p^N``. This makes identical rankings score
    1 and their corresponding distance 0.

    ``persistence`` must be in ``[0, 1)``. Larger values put more weight on
    lower-ranked items.
    """
    _validate_ranking_pair(true_ranking, predicted_ranking)
    if not 0.0 <= persistence < 1.0:
        raise ValueError("RBO persistence must satisfy 0 <= p < 1.")

    max_depth = max(len(true_ranking), len(predicted_ranking))
    true_prefix = set()
    predicted_prefix = set()
    overlap = 0
    weighted_overlap = 0.0

    for depth in range(1, max_depth + 1):
        if depth <= len(true_ranking):
            item_id = true_ranking[depth - 1]
            if item_id not in true_prefix:
                true_prefix.add(item_id)
                if item_id in predicted_prefix:
                    overlap += 1
        if depth <= len(predicted_ranking):
            item_id = predicted_ranking[depth - 1]
            if item_id not in predicted_prefix:
                predicted_prefix.add(item_id)
                if item_id in true_prefix:
                    overlap += 1

        weighted_overlap += (
            persistence ** (depth - 1) * overlap / depth
        )

    raw_similarity = (1.0 - persistence) * weighted_overlap
    identical_similarity = 1.0 - persistence ** max_depth
    similarity = raw_similarity / identical_similarity
    similarity = min(1.0, max(0.0, similarity))
    if abs(similarity - 1.0) < 1e-12:
        return 1.0
    if abs(similarity) < 1e-12:
        return 0.0
    return similarity


def rbo_distance(
    true_ranking: Sequence[int],
    predicted_ranking: Sequence[int],
    persistence: float,
) -> float:
    """Return ``1 - RBO similarity``."""
    return 1.0 - rbo_similarity(
        true_ranking,
        predicted_ranking,
        persistence,
    )
