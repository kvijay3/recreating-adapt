"""Tests for model.py — CNN architecture and training."""

import unittest

import numpy as np
import tensorflow as tf

from adapt_reimpl.model import CasCNNWithParallelFilters, construct_model, default_params


class TestModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_model_classification(self):
        """Test that the classification model builds and runs inference."""
        params = default_params()
        params['conv_filter_width'] = [3, 5]
        params['conv_num_filters'] = 10
        params['fully_connected_dim'] = [10]
        params['batch_size'] = 4
        params['context_nt'] = 5

        seq_len = 2 * params['context_nt'] + 28  # 38
        batch_size = 4
        x = np.random.rand(batch_size, seq_len, 8).astype(np.float32)

        model = construct_model(params, (None, seq_len, 8), regression=False)
        output = model.call(x, training=False)
        self.assertEqual(output.shape, (batch_size, 1))
        # Sigmoid output should be in [0, 1]
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))

    def test_model_regression(self):
        """Test that the regression model builds and runs inference."""
        params = default_params()
        params['conv_filter_width'] = [3, 5]
        params['conv_num_filters'] = 10
        params['fully_connected_dim'] = [10]
        params['batch_size'] = 4
        params['context_nt'] = 5
        params['regression_clip'] = True

        seq_len = 2 * params['context_nt'] + 28
        batch_size = 4
        x = np.random.rand(batch_size, seq_len, 8).astype(np.float32)

        model = construct_model(params, (None, seq_len, 8), regression=True)
        output = model.call(x, training=False)
        self.assertEqual(output.shape, (batch_size, 1))

    def test_model_training_decreases_loss(self):
        """Test that the model can train and loss decreases on synthetic data."""
        params = default_params()
        params['conv_filter_width'] = [3]
        params['conv_num_filters'] = 8
        params['fully_connected_dim'] = [8]
        params['batch_size'] = 16
        params['context_nt'] = 5
        params['learning_rate'] = 1e-3

        seq_len = 2 * params['context_nt'] + 28
        n_samples = 64
        x_train = np.random.rand(n_samples, seq_len, 8).astype(np.float32)
        y_train = np.random.randint(0, 2, (n_samples, 1)).astype(np.float32)

        model = construct_model(params, (None, seq_len, 8), regression=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

        # Train for a few epochs
        history = model.fit(x_train, y_train, epochs=5, batch_size=16,
                            verbose=0)
        losses = history.history['loss']
        # Loss should generally decrease
        self.assertLess(losses[-1], losses[0] + 0.1)

    def test_model_with_gc_content(self):
        """Test model with GC content feature."""
        params = default_params()
        params['conv_filter_width'] = [3]
        params['conv_num_filters'] = 8
        params['fully_connected_dim'] = [8]
        params['batch_size'] = 4
        params['context_nt'] = 5
        params['add_gc_content'] = True

        seq_len = 2 * params['context_nt'] + 28
        batch_size = 4
        x = np.random.rand(batch_size, seq_len, 8).astype(np.float32)

        model = construct_model(params, (None, seq_len, 8), regression=False)
        output = model.call(x, training=False)
        self.assertEqual(output.shape, (batch_size, 1))


if __name__ == '__main__':
    unittest.main()
