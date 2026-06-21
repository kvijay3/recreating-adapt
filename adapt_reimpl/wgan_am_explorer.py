"""WGAN-AM explorer for BADGERS.

Uses a trained WGAN generator to produce guide sequences, then applies
adversarial mismatches to optimize guides for diagnostic fitness.
"""

import numpy as np

from adapt_reimpl.sequence_utils import convert_to_nt, one_hot_encode_base
from adapt_reimpl.gan import WGAN


class WGANAMExplorer:
    """WGAN with Adversarial Mismatches explorer.

    Generates candidate guides from a WGAN, then introduces artificial
    mismatches to improve diagnostic specificity.
    """

    def __init__(self, fitness_model, wgan=None, guide_length=28,
                 noise_dim=100, num_candidates=200,
                 mismatch_positions=None, max_mismatches=3,
                 seed=None):
        """Initialize the WGAN-AM explorer.

        Args:
            fitness_model: object with get_fitness(guide_seqs) method
            wgan: pre-trained WGAN object (None to create new)
            guide_length: length of guide sequences
            noise_dim: dimension of noise vector
            num_candidates: number of candidates to generate per round
            mismatch_positions: positions to consider for mismatches (None = all)
            max_mismatches: maximum number of artificial mismatches
            seed: random seed
        """
        self.fitness_model = fitness_model
        self.guide_length = guide_length
        self.noise_dim = noise_dim
        self.num_candidates = num_candidates
        self.max_mismatches = max_mismatches
        self.mismatch_positions = (
            mismatch_positions if mismatch_positions is not None
            else list(range(guide_length)))

        if seed is not None:
            np.random.seed(seed)

        if wgan is not None:
            self.wgan = wgan
        else:
            self.wgan = WGAN(
                guide_length=guide_length, noise_dim=noise_dim)

        self.best_guide = None
        self.best_fitness = -float('inf')
        self.fitness_history = []

    def _onehot_to_seq(self, onehot):
        """Convert one-hot encoded guide to nucleotide string.

        Args:
            onehot: numpy array (guide_length, 4)

        Returns:
            nucleotide string
        """
        return convert_to_nt(onehot)

    def _generate_candidates(self, n):
        """Generate candidate guides from the WGAN.

        Args:
            n: number of candidates

        Returns:
            list of guide sequences
        """
        onehot_guides = self.wgan.generate(n)
        guides = []
        for i in range(n):
            seq = self._onehot_to_seq(onehot_guides[i])
            if len(seq) == self.guide_length and 'N' not in seq:
                guides.append(seq)
        return guides

    def _apply_mismatches(self, guide, num_mismatches):
        """Apply random artificial mismatches to a guide.

        Args:
            guide: guide sequence string
            num_mismatches: number of mismatches to introduce

        Returns:
            list of mismatched guide variants
        """
        bases = 'ACGT'
        variants = []
        positions = np.random.choice(
            self.mismatch_positions,
            size=min(num_mismatches, len(self.mismatch_positions)),
            replace=False)

        for pos in positions:
            for new_base in bases:
                if new_base == guide[pos]:
                    continue
                variant = guide[:pos] + new_base + guide[pos + 1:]
                variants.append(variant)
        return variants

    def _optimize_mismatches(self, guide):
        """Greedy mismatch optimization.

        Iteratively adds mismatches that improve fitness.

        Args:
            guide: starting guide sequence

        Returns:
            (best_guide, best_fitness) after mismatch optimization
        """
        best_guide = guide
        best_fitness = self.fitness_model.get_fitness([guide])[0]

        for num_mm in range(1, self.max_mismatches + 1):
            improved = False
            candidates = self._apply_mismatches(best_guide, num_mismatches)
            if not candidates:
                continue

            fitnesses = self.fitness_model.get_fitness(candidates)
            best_idx = np.argmax(fitnesses)

            if fitnesses[best_idx] > best_fitness:
                best_guide = candidates[best_idx]
                best_fitness = fitnesses[best_idx]
                improved = True

            if not improved:
                break

        return best_guide, best_fitness

    def run(self, num_rounds=10):
        """Run the WGAN-AM optimization.

        Args:
            num_rounds: number of optimization rounds

        Returns:
            (best_guide, best_fitness) tuple
        """
        for round_idx in range(num_rounds):
            # Generate candidates from WGAN
            candidates = self._generate_candidates(self.num_candidates)
            if not candidates:
                continue

            # Evaluate base fitness
            base_fitnesses = self.fitness_model.get_fitness(candidates)

            # Take top candidates and optimize mismatches
            top_indices = np.argsort(base_fitnesses)[-20:]
            for idx in top_indices:
                guide = candidates[idx]
                opt_guide, opt_fitness = self._optimize_mismatches(guide)

                if opt_fitness > self.best_fitness:
                    self.best_fitness = opt_fitness
                    self.best_guide = opt_guide

            self.fitness_history.append(self.best_fitness)
            print(f"  WGAN-AM Round {round_idx + 1}/{num_rounds}: "
                  f"best fitness = {self.best_fitness:.4f}")

        return self.best_guide, self.best_fitness

    def load_wgan(self, path):
        """Load pre-trained WGAN weights.

        Args:
            path: directory containing generator.weights and discriminator.weights
        """
        self.wgan.load(path)

    def train_wgan(self, real_guides, batch_size=64, num_epochs=100):
        """Train the WGAN on real guide sequences.

        Args:
            real_guides: list of guide sequences
            batch_size: training batch size
            num_epochs: number of training epochs
        """
        # Convert to one-hot
        onehot = np.zeros((len(real_guides), self.guide_length, 4),
                          dtype=np.float32)
        for i, guide in enumerate(real_guides):
            for j, base in enumerate(guide):
                v = one_hot_encode_base(base)
                onehot[i, j] = v

        self.wgan.train(onehot, batch_size=batch_size, num_epochs=num_epochs)
