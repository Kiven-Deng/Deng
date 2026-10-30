import unittest

import torch

from kemeny_transformer.model.mallows import MallowsPhiSampler
from kemeny_transformer.model.ranking_data_generator import (
    generate_ranking_dataset,
    generate_noisy_rankings,
    generate_true_rankings,
)


class RankingDataGeneratorTests(unittest.TestCase):
    def test_true_rankings_are_reproducible_valid_tables(self):
        first = generate_true_rankings(20, 9, seed=17)
        second = generate_true_rankings(20, 9, seed=17)

        self.assertEqual(tuple(first.shape), (20, 9))
        self.assertTrue(torch.equal(first, second))
        expected = torch.arange(9).expand_as(first)
        self.assertTrue(
            torch.equal(torch.sort(first, dim=1).values, expected)
        )

    def test_dataset_shapes_and_noise_are_reproducible(self):
        first = generate_ranking_dataset(
            6,
            7,
            num_noisy_rankings=4,
            theta=1.5,
            seed=23,
        )
        second = generate_ranking_dataset(
            6,
            7,
            num_noisy_rankings=4,
            theta=1.5,
            seed=23,
        )

        self.assertEqual(tuple(first.true_rankings.shape), (6, 7))
        self.assertEqual(tuple(first.noisy_rankings.shape), (6, 4, 7))
        self.assertEqual(tuple(first.noisy_positions.shape), (6, 7, 4))
        self.assertTrue(
            torch.equal(first.true_rankings, second.true_rankings)
        )
        self.assertTrue(
            torch.equal(first.noisy_rankings, second.noisy_rankings)
        )
        expected = torch.arange(7).expand_as(first.noisy_rankings)
        self.assertTrue(
            torch.equal(
                torch.sort(first.noisy_rankings, dim=-1).values,
                expected,
            )
        )

    def test_noise_can_be_added_to_existing_true_rankings(self):
        true_rankings = generate_true_rankings(8, 5, seed=31)
        first = generate_noisy_rankings(
            true_rankings,
            num_noisy_rankings=2,
            theta=0.8,
            seed=71,
        )
        second = generate_noisy_rankings(
            true_rankings,
            num_noisy_rankings=2,
            theta=0.8,
            seed=71,
        )

        self.assertTrue(
            torch.equal(first.true_rankings, true_rankings)
        )
        self.assertTrue(first.true_rankings.is_contiguous())
        self.assertTrue(torch.equal(first.noisy_rankings, second.noisy_rankings))
        self.assertEqual(tuple(first.noisy_positions.shape), (8, 5, 2))

    def test_position_matrix_uses_one_based_ranks_per_item(self):
        dataset = generate_ranking_dataset(
            5,
            6,
            num_noisy_rankings=3,
            theta=0.7,
            seed=11,
        )
        expected = torch.arange(1, 7).view(1, 6, 1).expand_as(
            dataset.noisy_positions
        ).to(dtype=torch.float32)
        self.assertEqual(dataset.noisy_positions.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                torch.sort(dataset.noisy_positions, dim=1).values,
                expected,
            )
        )

    def test_theta_controls_average_kendall_distance(self):
        diffuse_dataset = generate_ranking_dataset(
            30,
            8,
            num_noisy_rankings=20,
            theta=0.1,
            seed=3,
        )
        diffuse_distance = MallowsPhiSampler.kendall_distance(
            diffuse_dataset.noisy_rankings.reshape(-1, 8),
            diffuse_dataset.true_rankings.repeat_interleave(20, dim=0),
        ).float().mean()

        dataset = generate_ranking_dataset(
            30,
            8,
            num_noisy_rankings=20,
            theta=4.0,
            seed=3,
        )
        concentrated_distance = MallowsPhiSampler.kendall_distance(
            dataset.noisy_rankings.reshape(-1, 8),
            dataset.true_rankings.repeat_interleave(20, dim=0),
        ).float().mean()
        self.assertGreater(diffuse_distance, concentrated_distance)

    def test_invalid_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            generate_true_rankings(0, 5)
        with self.assertRaises(ValueError):
            generate_true_rankings(5, 0)
        with self.assertRaises(ValueError):
            generate_ranking_dataset(5, 5, num_noisy_rankings=0)
        with self.assertRaises(ValueError):
            generate_ranking_dataset(5, 5, theta=-0.1)


if __name__ == "__main__":
    unittest.main()
