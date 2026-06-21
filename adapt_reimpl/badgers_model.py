"""Fitness models for BADGERS.

Cas13Mult: multi-target detection (maximize expected activity across targets)
Cas13Diff: variant identification (maximize on-target while minimizing off-target)
"""

import numpy as np
from collections import defaultdict

from adapt_reimpl.sequence_utils import (
    one_hot_encode_base,
    determine_consensus,
    FASTA_CODES,
)


class Cas13Mult:
    """Fitness model for multi-target detection.

    Computes expected activity of a guide across a set of target sequences,
    weighted by target frequency.
    """

    def __init__(self, target_seqs, predictor, guide_length=28,
                 context_nt=10):
        """Initialize the model.

        Args:
            target_seqs: list of target sequences (48-nt each, with context)
            predictor: Predictor object (classifier + regressor)
            guide_length: length of guide
            context_nt: context nucleotides on each side
        """
        self.target_seqs = target_seqs
        self.predictor = predictor
        self.guide_length = guide_length
        self.context_nt = context_nt

        # Compute consensus and unique targets with frequencies
        self.consensus = determine_consensus(target_seqs)
        seq_counts = defaultdict(int)
        for s in target_seqs:
            seq_counts[s] += 1
        self.unique_targets = list(seq_counts.keys())
        self.target_freqs = np.array(
            [seq_counts[s] / len(target_seqs) for s in self.unique_targets])

        # Parent sequences for exploration: consensus + unique targets
        self.parent_seqs = [self.consensus] + self.unique_targets

    def get_fitness(self, guide_seqs):
        """Compute fitness for each guide sequence.

        For each guide:
        1. One-hot encode guide-target pairs
        2. Run classifier + regressor
        3. Combined activity = classifier_score * (regressor_score + 4) - 4
        4. Weight by target frequency → expected activity

        Args:
            guide_seqs: list of guide sequences (28-nt each)

        Returns:
            numpy array of fitness scores, one per guide
        """
        fitness_scores = np.zeros(len(guide_seqs))

        for i, guide in enumerate(guide_seqs):
            if len(guide) != self.guide_length:
                fitness_scores[i] = -4.0
                continue

            # Build pairs for all unique targets
            pairs = [(target, guide) for target in self.unique_targets]

            # Compute activities
            activities = self.predictor.compute_activity(0, pairs)
            activities = np.array(activities)

            # Weight by target frequency
            expected_activity = np.sum(activities * self.target_freqs)
            fitness_scores[i] = expected_activity

        return fitness_scores


class Cas13Diff:
    """Fitness model for variant identification.

    Fitness = -(weighted cost), where cost penalizes off-target activity
    and rewards on-target activity.
    """

    def __init__(self, on_target_seqs, off_target_seqs, predictor,
                 guide_length=28, context_nt=10,
                 c=1.0, a=1.0, k=1.0, o=0.0, t2w=1.0):
        """Initialize the model.

        Args:
            on_target_seqs: list of target sequences to detect
            off_target_seqs: list of targets to avoid detecting
            predictor: Predictor object
            guide_length: length of guide
            context_nt: context nucleotides
            c, a, k, o, t2w: sigmoid cost function hyperparameters
        """
        self.on_target_seqs = on_target_seqs
        self.off_target_seqs = off_target_seqs
        self.predictor = predictor
        self.guide_length = guide_length
        self.context_nt = context_nt
        self.c = c
        self.a = a
        self.k = k
        self.o = o
        self.t2w = t2w

        # Compute consensus for parents
        self.consensus = determine_consensus(on_target_seqs)
        self.parent_seqs = [self.consensus] + on_target_seqs[:5]

    def _sigmoid_cost(self, activity):
        """Sigmoid cost function.

        Args:
            activity: predicted activity value

        Returns:
            cost value
        """
        return self.c / (1 + np.exp(-self.k * (activity - self.o)))

    def get_fitness(self, guide_seqs):
        """Compute fitness for each guide (negative cost).

        Args:
            guide_seqs: list of guide sequences

        Returns:
            numpy array of fitness scores (higher = better)
        """
        fitness_scores = np.zeros(len(guide_seqs))

        for i, guide in enumerate(guide_seqs):
            if len(guide) != self.guide_length:
                fitness_scores[i] = -100.0
                continue

            # On-target activity (reward)
            on_pairs = [(t, guide) for t in self.on_target_seqs]
            on_activities = np.array(
                self.predictor.compute_activity(0, on_pairs))
            on_cost = np.mean(self._sigmoid_cost(on_activities))

            # Off-target activity (penalty)
            if len(self.off_target_seqs) > 0:
                off_pairs = [(t, guide) for t in self.off_target_seqs]
                off_activities = np.array(
                    self.predictor.compute_activity(0, off_pairs))
                off_cost = np.mean(self._sigmoid_cost(off_activities))
            else:
                off_cost = 0.0

            # Total cost = penalty for off-target + reward for on-target
            total_cost = self.a * off_cost - on_cost
            fitness_scores[i] = -total_cost

        return fitness_scores
