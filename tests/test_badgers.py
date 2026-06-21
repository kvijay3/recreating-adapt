"""Tests for BADGERS: evolutionary explorer, WGAN, and fitness models."""

import unittest

import numpy as np

from adapt_reimpl.sequence_utils import one_hot_encode_base, convert_to_nt
from adapt_reimpl.evolutionary_explorer import EvolutionaryExplorer
from adapt_reimpl.gan import WGAN, GuideGenerator, GuideDiscriminator


class MockFitnessModel:
    """Simple mock fitness model for testing."""

    def __init__(self, guide_length=28, target_guide=None):
        self.guide_length = guide_length
        self.target_guide = target_guide or ('ACGT' * 7)
        self.parent_seqs = [self.target_guide]

    def get_fitness(self, guide_seqs):
        from adapt_reimpl.sequence_utils import hamming_distance
        scores = np.zeros(len(guide_seqs))
        for i, guide in enumerate(guide_seqs):
            if len(guide) != self.guide_length:
                scores[i] = -100.0
            else:
                # Fitness = -hamming_distance to target
                scores[i] = -float(hamming_distance(guide, self.target_guide))
        return scores


class TestEvolutionaryExplorer(unittest.TestCase):

    def test_produces_valid_guides(self):
        """Verify evolutionary explorer produces valid 28-mer guides."""
        fitness_model = MockFitnessModel()
        explorer = EvolutionaryExplorer(
            fitness_model, guide_length=28,
            population_size=20, num_generations=10,
            mutation_rate=0.05, seed=42)
        best_guide, best_fitness = explorer.run()
        self.assertEqual(len(best_guide), 28)
        self.assertTrue(all(c in 'ACGT' for c in best_guide))

    def test_fitness_improves(self):
        """Verify fitness improves over generations."""
        fitness_model = MockFitnessModel()
        explorer = EvolutionaryExplorer(
            fitness_model, guide_length=28,
            population_size=30, num_generations=20,
            mutation_rate=0.02, seed=42)
        explorer.run()
        # Fitness history should show improvement
        self.assertGreater(len(explorer.fitness_history), 0)
        self.assertGreaterEqual(
            explorer.fitness_history[-1], explorer.fitness_history[0])

    def test_converges_to_target(self):
        """Verify the explorer can converge to a known target."""
        target = 'AAAA' * 7  # 28-mer of all A's
        fitness_model = MockFitnessModel(target_guide=target)
        explorer = EvolutionaryExplorer(
            fitness_model, guide_length=28,
            population_size=50, num_generations=50,
            mutation_rate=0.01, seed=42)
        best_guide, best_fitness = explorer.run()
        # Should get reasonably close to target
        self.assertGreater(best_fitness, -10.0)  # < 10 mismatches


class TestWGAN(unittest.TestCase):

    def test_generator_output_shape(self):
        """Verify generator produces correct output shape."""
        gen = GuideGenerator(noise_dim=50, guide_length=28)
        noise = np.random.randn(4, 50).astype(np.float32)
        import tensorflow as tf
        output = gen(tf.convert_to_tensor(noise))
        self.assertEqual(output.shape, (4, 28, 4))
        # Softmax output should sum to 1 per position
        sums = np.sum(output.numpy(), axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_discriminator_output_shape(self):
        """Verify discriminator produces correct output shape."""
        disc = GuideDiscriminator(guide_length=28)
        x = np.random.rand(4, 28, 4).astype(np.float32)
        import tensorflow as tf
        output = disc(tf.convert_to_tensor(x))
        self.assertEqual(output.shape, (4, 1))

    def test_wgan_generate(self):
        """Verify WGAN can generate guides."""
        wgan = WGAN(guide_length=28, noise_dim=50)
        # Build models
        import tensorflow as tf
        dummy = tf.zeros([1, 50])
        _ = wgan.generator(dummy)
        _ = wgan.discriminator(tf.zeros([1, 28, 4]))

        guides = wgan.generate(5)
        self.assertEqual(guides.shape, (5, 28, 4))

    def test_wgan_save_load(self):
        """Verify WGAN can save and load weights."""
        import os
        import tempfile
        import tensorflow as tf

        wgan = WGAN(guide_length=28, noise_dim=50)
        # Build models
        dummy = tf.zeros([1, 50])
        _ = wgan.generator(dummy)
        _ = wgan.discriminator(tf.zeros([1, 28, 4]))

        with tempfile.TemporaryDirectory() as tmpdir:
            wgan.save(tmpdir)
            self.assertTrue(os.path.exists(
                os.path.join(tmpdir, 'generator.weights.h5')))

            wgan2 = WGAN(guide_length=28, noise_dim=50)
            _ = wgan2.generator(tf.zeros([1, 50]))
            _ = wgan2.discriminator(tf.zeros([1, 28, 4]))
            wgan2.load(tmpdir)

            # Generate from both and check shapes match
            g1 = wgan.generate(2)
            g2 = wgan2.generate(2)
            self.assertEqual(g1.shape, g2.shape)


class TestWGANAMExplorer(unittest.TestCase):

    def test_initialization(self):
        """Verify WGAN-AM explorer initializes correctly."""
        from adapt_reimpl.wgan_am_explorer import WGANAMExplorer
        fitness_model = MockFitnessModel()
        explorer = WGANAMExplorer(
            fitness_model, guide_length=28,
            num_candidates=10, max_mismatches=2, seed=42)
        self.assertEqual(explorer.guide_length, 28)
        self.assertEqual(explorer.max_mismatches, 2)


if __name__ == '__main__':
    unittest.main()
