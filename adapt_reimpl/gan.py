"""Wasserstein GAN for generating Cas13a guide sequences.

Generator: ResNet-based, takes noise vector → one-hot encoded guide sequence.
Discriminator: CNN-based, classifies real vs generated guide sequences.
"""

import numpy as np
import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    """Residual block for the generator network."""

    def __init__(self, units, activation='relu', **kwargs):
        """Initialize residual block.

        Args:
            units: number of hidden units
            activation: activation function name
        """
        super(ResidualBlock, self).__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(units, activation=activation)
        self.fc2 = tf.keras.layers.Dense(units)
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, x, training=False):
        """Forward pass.

        Args:
            x: input tensor
            training: whether in training mode

        Returns:
            output tensor
        """
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + residual
        return self.activation(x)


class GuideGenerator(tf.keras.Model):
    """Generator network for guide sequences.

    Takes a noise vector and produces a one-hot encoded guide sequence.
    Uses ResNet-style architecture.
    """

    def __init__(self, noise_dim=100, guide_length=28,
                 hidden_dim=256, num_res_blocks=3):
        """Initialize the generator.

        Args:
            noise_dim: dimension of input noise vector
            guide_length: length of guide sequence
            hidden_dim: hidden layer dimension
            num_res_blocks: number of residual blocks
        """
        super(GuideGenerator, self).__init__()
        self.guide_length = guide_length
        self.noise_dim = noise_dim

        self.fc_initial = tf.keras.layers.Dense(hidden_dim)
        self.res_blocks = [
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        self.fc_out = tf.keras.layers.Dense(guide_length * 4)
        self.reshape = tf.keras.layers.Reshape((guide_length, 4))
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x, training=False):
        """Forward pass.

        Args:
            x: noise vector of shape (batch, noise_dim)
            training: whether in training mode

        Returns:
            one-hot encoded guide of shape (batch, guide_length, 4)
        """
        x = self.fc_initial(x)
        for block in self.res_blocks:
            x = block(x, training=training)
        x = self.fc_out(x)
        x = self.reshape(x)
        return self.softmax(x)


class GuideDiscriminator(tf.keras.Model):
    """Discriminator network for guide sequences.

    Classifies one-hot encoded guide sequences as real or generated.
    """

    def __init__(self, guide_length=28, hidden_dim=128):
        """Initialize the discriminator.

        Args:
            guide_length: length of guide sequence
            hidden_dim: hidden layer dimension
        """
        super(GuideDiscriminator, self).__init__()
        self.guide_length = guide_length

        self.conv1 = tf.keras.layers.Conv1D(
            32, 3, strides=1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(
            64, 3, strides=2, padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x, training=False):
        """Forward pass.

        Args:
            x: one-hot encoded guide of shape (batch, guide_length, 4)
            training: whether in training mode

        Returns:
            critic scores of shape (batch, 1)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)


class WGAN:
    """Wasserstein GAN for guide sequence generation.

    Uses WGAN-GP (gradient penalty) training for stability.
    """

    def __init__(self, guide_length=28, noise_dim=100,
                 generator_lr=1e-4, discriminator_lr=1e-4,
                 lambda_gp=10.0, n_critic=5):
        """Initialize the WGAN.

        Args:
            guide_length: length of guide sequence
            noise_dim: dimension of noise vector
            generator_lr: generator learning rate
            discriminator_lr: discriminator learning rate
            lambda_gp: gradient penalty coefficient
            n_critic: number of discriminator updates per generator update
        """
        self.guide_length = guide_length
        self.noise_dim = noise_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

        self.generator = GuideGenerator(
            noise_dim=noise_dim, guide_length=guide_length)
        self.discriminator = GuideDiscriminator(guide_length=guide_length)

        self.gen_optimizer = tf.keras.optimizers.Adam(
            generator_lr, beta_1=0.5, beta_2=0.9)
        self.disc_optimizer = tf.keras.optimizers.Adam(
            discriminator_lr, beta_1=0.5, beta_2=0.9)

    def _gradient_penalty(self, real_data, fake_data):
        """Compute gradient penalty for WGAN-GP.

        Args:
            real_data: real guide sequences (batch, guide_length, 4)
            fake_data: generated guide sequences (batch, guide_length, 4)

        Returns:
            gradient penalty scalar
        """
        batch_size = tf.shape(real_data)[0]
        eps = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = eps * real_data + (1 - eps) * fake_data

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = tape.gradient(pred, interpolated)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-8)
        penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)
        return penalty

    def _discriminator_loss(self, real_data, fake_data):
        """Wasserstein discriminator loss.

        Args:
            real_data: real guide sequences
            fake_data: generated guide sequences

        Returns:
            loss scalar
        """
        real_scores = self.discriminator(real_data, training=True)
        fake_scores = self.discriminator(fake_data, training=True)
        gp = self._gradient_penalty(real_data, fake_data)
        loss = -tf.reduce_mean(real_scores) + tf.reduce_mean(fake_scores) + self.lambda_gp * gp
        return loss

    def _generator_loss(self, fake_data):
        """Wasserstein generator loss.

        Args:
            fake_data: generated guide sequences

        Returns:
            loss scalar
        """
        fake_scores = self.discriminator(fake_data, training=False)
        return -tf.reduce_mean(fake_scores)

    @tf.function
    def train_disc_step(self, real_data):
        """Single discriminator training step.

        Args:
            real_data: real guide sequences (batch, guide_length, 4)

        Returns:
            discriminator loss
        """
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            loss = self._discriminator_loss(real_data, fake_data)

        grads = tape.gradient(
            loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables))
        return loss

    @tf.function
    def train_gen_step(self, batch_size):
        """Single generator training step.

        Args:
            batch_size: batch size

        Returns:
            generator loss
        """
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            loss = self._generator_loss(fake_data)

        grads = tape.gradient(
            loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))
        return loss

    def train(self, real_guides_onehot, batch_size=64, num_epochs=100,
              verbose=True):
        """Train the WGAN.

        Args:
            real_guides_onehot: numpy array (n, guide_length, 4) of real guides
            batch_size: training batch size
            num_epochs: number of training epochs
            verbose: if True, print progress
        """
        n_samples = len(real_guides_onehot)
        steps_per_epoch = n_samples // batch_size

        for epoch in range(num_epochs):
            for step in range(steps_per_epoch):
                # Train discriminator n_critic times
                for _ in range(self.n_critic):
                    idx = np.random.randint(0, n_samples, batch_size)
                    real_batch = real_guides_onehot[idx]
                    self.train_disc_step(
                        tf.convert_to_tensor(real_batch, dtype=tf.float32))

                # Train generator
                self.train_gen_step(batch_size)

            if verbose and epoch % 10 == 0:
                print(f"  WGAN Epoch {epoch}/{num_epochs}")

    def generate(self, n, noise=None):
        """Generate guide sequences.

        Args:
            n: number of guides to generate
            noise: optional noise vector (n, noise_dim)

        Returns:
            numpy array of one-hot encoded guides (n, guide_length, 4)
        """
        if noise is None:
            noise = np.random.randn(n, self.noise_dim).astype(np.float32)
        generated = self.generator(
            tf.convert_to_tensor(noise), training=False)
        return generated.numpy()

    def save(self, path):
        """Save generator and discriminator weights.

        Args:
            path: directory to save to
        """
        import os
        os.makedirs(path, exist_ok=True)
        self.generator.save_weights(os.path.join(path, 'generator.weights.h5'))
        self.discriminator.save_weights(
            os.path.join(path, 'discriminator.weights.h5'))

    def load(self, path):
        """Load generator and discriminator weights.

        Args:
            path: directory to load from
        """
        import os
        # Build models first
        dummy_noise = tf.zeros([1, self.noise_dim])
        _ = self.generator(dummy_noise)
        dummy_input = tf.zeros([1, self.guide_length, 4])
        _ = self.discriminator(dummy_input)
        self.generator.load_weights(os.path.join(path, 'generator.weights.h5'))
        self.discriminator.load_weights(
            os.path.join(path, 'discriminator.weights.h5'))
