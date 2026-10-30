import copy
import unittest

import torch
import torch.nn as nn

from kemeny_transformer.model.architecture import (
    EmbeddingType,
    IdentityEmbedding,
    RotaryRankBias,
    calculate_consensus_positions,
    kemeny_transformer,
)
from kemeny_transformer.model.mallows import MallowsPhiSampler
from kemeny_transformer.model.ranking_data_generator import (
    generate_ranking_dataset,
)


def tiny_model(**overrides):
    config = dict(
        embedding_type=EmbeddingType.MALLOWS_IDENTITY,
        input_dim=8,
        embedding_dim=8,
        dim_ff=16,
        numb_heads=2,
        numb_layers_decoder=2,
        numb_layers_encoder=1,
        max_len_PE=16,
        num_generated_rankings=8,
        mallows_theta=1.0,
        mallows_seed=7,
    )
    config.update(overrides)
    return kemeny_transformer(**config)


class MallowsSamplingTests(unittest.TestCase):
    def test_shape_reproducibility_and_valid_permutations(self):
        center = torch.tensor([[0, 1, 2, 3, 4], [4, 2, 0, 3, 1]])
        first = MallowsPhiSampler(theta=0.8, seed=123)
        second = MallowsPhiSampler(theta=0.8, seed=123)
        orders_a, d_a = first.sample_rankings(center, 7)
        orders_b, d_b = second.sample_rankings(center, 7)

        self.assertEqual(orders_a.shape, (2, 7, 5))
        self.assertEqual(d_a, 7)
        self.assertEqual(d_b, 7)
        self.assertTrue(torch.equal(orders_a, orders_b))
        expected = torch.arange(5).expand_as(orders_a)
        self.assertTrue(torch.equal(torch.sort(orders_a, dim=-1).values, expected))

        positions = first.rankings_to_position_matrix(orders_a, rank_base=1)
        self.assertEqual(positions.shape, (2, 5, 7))
        for batch in range(2):
            for ranking in range(7):
                self.assertEqual(
                    sorted(positions[batch, :, ranking].tolist()),
                    [1, 2, 3, 4, 5],
                )

    def test_theta_controls_average_kendall_distance(self):
        center = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        diffuse, _ = MallowsPhiSampler(0.1, seed=9).sample_rankings(center, 2000)
        concentrated, _ = MallowsPhiSampler(
            4.0, seed=9
        ).sample_rankings(center, 2000)
        diffuse_distance = MallowsPhiSampler.kendall_distance(
            diffuse[:, 0], center
        ).float().mean()
        concentrated_distance = MallowsPhiSampler.kendall_distance(
            concentrated[:, 0], center
        ).float().mean()
        self.assertGreater(diffuse_distance, concentrated_distance)
        self.assertEqual(
            MallowsPhiSampler.kendall_distance(center, center).item(), 0
        )

    def test_invalid_center_ranking_is_rejected(self):
        with self.assertRaises(ValueError):
            MallowsPhiSampler(seed=0).sample_position_matrix(
                torch.tensor([[0, 0, 2]]), 4
            )


class ModelArchitectureTests(unittest.TestCase):
    def test_identity_has_no_projection_and_shape_is_preserved(self):
        embedding = IdentityEmbedding(8)
        x = torch.randn(3, 5, 8)
        self.assertEqual(len(list(embedding.parameters())), 0)
        self.assertTrue(torch.equal(embedding(x), x))
        with self.assertRaises(ValueError):
            embedding(torch.randn(3, 5, 7))

    def test_num_rankings_must_equal_embedding_dim(self):
        with self.assertRaises(ValueError):
            tiny_model(num_generated_rankings=7)

    def test_default_target_layers_are_pre_batchnorm(self):
        model = tiny_model()
        self.assertEqual(model.norm_type, "batchnorm")
        self.assertEqual(model.norm_position, "pre")
        self.assertFalse(model.use_original_decoder_pe)
        self.assertIsInstance(
            model.encoder.norm1_layers[0],
            nn.BatchNorm1d,
        )
        self.assertEqual(len(model.decoder.decoder_layers), 1)
        decoder_layer = model.decoder.decoder_layers[0]
        self.assertIsInstance(decoder_layer.BN_selfatt, nn.BatchNorm1d)
        self.assertIsInstance(decoder_layer.BN_att, nn.BatchNorm1d)
        self.assertIsInstance(decoder_layer.BN_MLP, nn.BatchNorm1d)
        self.assertFalse(
            any(
                isinstance(module, nn.LayerNorm)
                for _, module in model.named_modules()
            )
        )

    def test_ffns_use_swiglu_without_changing_hidden_dimensions(self):
        model = tiny_model()
        encoder_ffn = model.encoder.linear1_layers[0]
        encoder_output = model.encoder.linear2_layers[0]
        decoder_ffn = model.decoder.decoder_layers[0].W1_MLP
        decoder_output = model.decoder.decoder_layers[0].W2_MLP

        # SwiGLU packs gate and value projections into one input projection.
        self.assertEqual(encoder_ffn.out_features, 2 * model.dim_ff)
        self.assertEqual(encoder_output.in_features, model.dim_ff)
        self.assertEqual(decoder_ffn.out_features, 2 * model.dim_emb)
        self.assertEqual(decoder_output.in_features, model.dim_emb)

    def test_rarope_preserves_qk_norm_and_start_has_zero_bias(self):
        position = RotaryRankBias(
            head_dim=4,
            numb_heads=2,
            use_rarope=True,
            use_anlibi=True,
            rarope_a=1.0,
            rarope_b=1.0,
            anlibi_alpha_init=0.01,
            anlibi_beta_init=0.02,
            anlibi_gamma=1.5,
        )
        q = torch.randn(1, 2, 4)
        k = torch.randn(1, 3, 4)
        q_positions = torch.tensor([[0, 1]])
        k_positions = torch.tensor([[0, 1, 3]])
        q_rotated, k_rotated = position.transform_qk(
            q, k, q_positions, k_positions
        )
        self.assertTrue(
            torch.allclose(q_rotated[:, 0], q[:, 0], atol=1.0e-6)
        )
        self.assertTrue(
            torch.allclose(k_rotated[:, 0], k[:, 0], atol=1.0e-6)
        )
        self.assertTrue(
            torch.allclose(
                q_rotated.norm(dim=-1),
                q.norm(dim=-1),
                atol=1.0e-5,
            )
        )
        bias = position.attention_bias(q_positions, k_positions)
        self.assertEqual(tuple(bias.shape), (1, 2, 2, 3))
        self.assertTrue(torch.equal(bias[..., 0], torch.zeros_like(bias[..., 0])))
        self.assertTrue(torch.equal(bias[:, :, 0, :], torch.zeros_like(bias[:, :, 0, :])))
        self.assertTrue((position.alpha >= 0).all())
        self.assertTrue((position.beta >= 0).all())

    def test_consensus_is_not_derived_from_target(self):
        noisy = torch.tensor(
            [[[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [0.0, 0.0]]]
        )
        padding = torch.tensor([[False, False, False, True]])
        positions = calculate_consensus_positions(noisy, padding)
        self.assertEqual(positions.tolist(), [[1, 2, 3, 0]])

    def test_start_placeholder_is_appended_after_identity_embedding(self):
        model = tiny_model()
        targets = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        noisy = model.generate_mallows_inputs(targets)
        padding = torch.zeros(2, 4, dtype=torch.bool)
        captured = []
        handle = model.encoder.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        model.teacher_forcing_forward(noisy, targets, padding)
        handle.remove()
        self.assertEqual(tuple(captured[0].shape), (2, 5, 8))
        self.assertTrue(
            torch.allclose(
                captured[0][:, -1],
                model.start_placeholder.expand(2, -1),
                atol=1.0e-6,
            )
        )

    def test_padding_does_not_change_encoder_bn_statistics(self):
        model_a = tiny_model()
        model_b = copy.deepcopy(model_a)
        mask = torch.tensor(
            [[False, False, False, True], [False, False, False, False]]
        )
        x_a = torch.randn(2, 4, 8)
        x_b = x_a.clone()
        x_b[0, 3] = 1.0e6
        model_a.encoder(x_a, mask)
        model_b.encoder(x_b, mask)

        # The final update uses only valid tokens in the final encoder sublayer.
        self.assertTrue(
            torch.allclose(
                model_a.encoder.norm2_layers[0].running_mean,
                model_b.encoder.norm2_layers[0].running_mean,
                atol=1.0e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                model_a.encoder.norm2_layers[0].running_var,
                model_b.encoder.norm2_layers[0].running_var,
                atol=1.0e-5,
            )
        )

    def test_teacher_forcing_uses_true_history_and_ce(self):
        model = tiny_model()
        model.train()
        targets = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
        noisy = model.generate_mallows_inputs(targets)
        padding = torch.zeros_like(targets, dtype=torch.bool)

        history = []
        original_next_input = model._next_decoder_input

        def record_history(h_encoder, item_indices, step):
            history.append(item_indices.detach().clone())
            return original_next_input(h_encoder, item_indices, step)

        model._next_decoder_input = record_history
        output = model.teacher_forcing_forward(noisy, targets, padding)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(len(history), 4)
        for step, recorded in enumerate(history):
            self.assertTrue(torch.equal(recorded, targets[:, step]))

        output["loss"].backward()
        encoder_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.encoder.parameters()
            if parameter.grad is not None
        )
        decoder_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.decoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_grad, 0)
        self.assertGreater(decoder_grad, 0)

    def test_dataset_interface_uses_noisy_encoder_and_true_decoder_inputs(self):
        model = tiny_model()
        model.train()
        dataset = generate_ranking_dataset(
            3,
            4,
            num_noisy_rankings=8,
            theta=1.2,
            seed=19,
        )

        history = []
        captured_encoder_input = []
        original_next_input = model._next_decoder_input

        def record_history(h_encoder, item_indices, step):
            history.append(item_indices.detach().clone())
            return original_next_input(h_encoder, item_indices, step)

        model._next_decoder_input = record_history
        handle = model.encoder.register_forward_pre_hook(
            lambda _module, inputs: captured_encoder_input.append(
                inputs[0][:, :-1].detach().clone()
            )
        )
        output = model.training_forward_from_dataset(
            {
                "noisy_positions": dataset.noisy_positions,
                "true_rankings": dataset.true_rankings,
            }
        )
        handle.remove()

        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(tuple(output["rankings"].shape), (3, 4))
        self.assertTrue(
            torch.equal(
                captured_encoder_input[0],
                dataset.noisy_positions,
            )
        )
        self.assertEqual(len(history), 4)
        for step, recorded in enumerate(history):
            self.assertTrue(
                torch.equal(recorded, dataset.true_rankings[:, step])
            )

    def test_dataset_interface_rejects_wrong_noise_dimension(self):
        model = tiny_model()
        dataset = generate_ranking_dataset(
            2,
            4,
            num_noisy_rankings=7,
            theta=1.0,
            seed=5,
        )
        with self.assertRaises(ValueError):
            model.training_forward_from_dataset(
                dataset.noisy_positions,
                dataset.true_rankings,
            )

    def test_future_target_does_not_change_earlier_log_probability(self):
        model = tiny_model()
        model.eval()
        other = copy.deepcopy(model)
        targets_a = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3]])
        targets_b = torch.tensor([[0, 2, 3, 1], [0, 3, 2, 1]])
        noisy = torch.arange(4).view(1, 4, 1).repeat(2, 1, 8).float()
        padding = torch.zeros(2, 4, dtype=torch.bool)

        output_a = model.teacher_forcing_forward(noisy, targets_a, padding)
        output_b = other.teacher_forcing_forward(noisy, targets_b, padding)
        self.assertTrue(
            torch.allclose(
                output_a["step_log_probs"][:, 0],
                output_b["step_log_probs"][:, 0],
                atol=1.0e-6,
            )
        )

    def test_variable_length_batch_and_batch_size_one_inference(self):
        model = tiny_model()
        targets = torch.tensor([[0, 2, 3, 1, 4], [4, 3, 2, 1, 0]])
        padding = torch.zeros(2, 5, dtype=torch.bool)
        padding[0, 4] = True
        noisy = model.generate_mallows_inputs(targets, padding_mask=padding)
        self.assertTrue(torch.equal(noisy[0, 4], torch.zeros(8)))
        output = model.teacher_forcing_forward(noisy, targets, padding)
        self.assertEqual(
            output["output_mask"].tolist(),
            [[False, False, False, False, True], [False, False, False, False, False]],
        )

        model.eval()
        ranking, _, output_mask = model(
            noisy[:1], padding[:1], deterministic=True
        )
        self.assertEqual(ranking.shape, (1, 5))
        self.assertEqual(int((~output_mask[0]).sum()), 4)
        valid = ranking[0, ~output_mask[0]] - 1
        self.assertEqual(sorted(valid.tolist()), [0, 1, 2, 3])

    def test_running_statistics_and_small_batch_overfit(self):
        torch.manual_seed(2)
        targets = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        )
        padding = torch.zeros(4, 4, dtype=torch.bool)
        model = tiny_model(
            mallows_theta=2.0,
            mallows_seed=5,
            dim_ff=32,
            numb_heads=4,
            numb_layers_decoder=3,
        )
        noisy = model.generate_mallows_inputs(targets)
        optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)
        losses = []
        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            output = model.teacher_forcing_forward(noisy, targets, padding)
            self.assertTrue(torch.isfinite(output["loss"]))
            output["loss"].backward()
            optimizer.step()
            losses.append(output["loss"].item())
        self.assertLess(losses[-1], losses[0] * 0.1)
        self.assertGreater(
            model.encoder.norm1_layers[0].num_batches_tracked.item(), 0
        )

        model.eval()
        ranking, _, output_mask = model(
            noisy, padding, deterministic=True
        )
        self.assertFalse(output_mask.any())
        self.assertTrue(
            all(
                torch.equal(torch.sort(row).values, torch.arange(4))
                for row in ranking - 1
            )
        )


if __name__ == "__main__":
    unittest.main()
