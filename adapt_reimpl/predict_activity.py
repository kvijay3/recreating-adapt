"""Predictor class for computing Cas13a guide activity.

Combines a classification model (active/inactive) with a regression model
(activity score on active pairs) in a hurdle model.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import numpy as np
import tensorflow as tf

from adapt_reimpl.sequence_utils import one_hot_encode_base
from adapt_reimpl.thermo import calculate_melting_temp


def _onehot(b):
    """One-hot encode a base (supports IUPAC codes)."""
    from adapt_reimpl.sequence_utils import FASTA_CODES, ONEHOT_IDX
    real_bases = FASTA_CODES.get(b, {b})
    v = [0.0, 0.0, 0.0, 0.0]
    for b_real in real_bases:
        if b_real in ONEHOT_IDX:
            v[ONEHOT_IDX[b_real]] = 1.0 / len(real_bases)
    return v


class BasePredictor:
    """Base class for predictors."""

    def __init__(self):
        pass

    def compute_activity(self, start_pos, pairs, percentiles=None):
        """Compute activity for guide-target pairs.

        Args:
            start_pos: start position (for memoization)
            pairs: list of (target_with_context, guide) tuples
            percentiles: optional percentile(s) to compute

        Returns:
            list of activity values, or (activities, percentiles) tuple
        """
        raise NotImplementedError

    def cleanup_memoized(self, start_pos):
        """Clean up memoized results for a start position."""
        if start_pos in self._memoized_evaluations:
            del self._memoized_evaluations[start_pos]


class Predictor(BasePredictor):
    """CNN-based predictor combining classifier and regressor.

    Uses a two-stage hurdle model:
    1. Classifier determines if a guide-target pair is active
    2. Regressor predicts activity score for active pairs
    """

    def __init__(self, classification_model_path, regression_model_path,
                 classification_threshold=None, regression_threshold=None):
        """Initialize the predictor.

        Args:
            classification_model_path: path to saved classification model
            regression_model_path: path to saved regression model
            classification_threshold: threshold for active classification;
                if None, read from model assets
            regression_threshold: threshold for highly active;
                if None, read from model assets
        """
        # Load models from saved weights + params
        import pickle
        from adapt_reimpl.model import construct_model

        cls_weights_file = os.path.join(
            classification_model_path, 'model.weights.h5')
        reg_weights_file = os.path.join(
            regression_model_path, 'model.weights.h5')

        cls_params_file = os.path.join(
            classification_model_path, 'model.params.pkl')
        reg_params_file = os.path.join(
            regression_model_path, 'model.params.pkl')

        with open(cls_params_file, 'rb') as f:
            cls_params = pickle.load(f)
        with open(reg_params_file, 'rb') as f:
            reg_params = pickle.load(f)

        # Reconstruct and load weights
        self.classification_model = construct_model(
            cls_params, (None, 2 * cls_params['context_nt'] + 28, 8),
            regression=False)
        self.classification_model.load_weights(cls_weights_file)

        self.regression_model = construct_model(
            reg_params, (None, 2 * reg_params['context_nt'] + 28, 8),
            regression=True)
        self.regression_model.load_weights(reg_weights_file)

        self.regression_lower_bound = 0.0
        self.regression_shift = 4.0
        self.rough_max_activity = 4.0

        # Load context_nt
        cls_ctx_path = os.path.join(
            classification_model_path, 'assets.extra', 'context_nt.arg')
        reg_ctx_path = os.path.join(
            regression_model_path, 'assets.extra', 'context_nt.arg')
        with open(cls_ctx_path) as f:
            cls_ctx = int(f.readline().strip())
        with open(reg_ctx_path) as f:
            reg_ctx = int(f.readline().strip())
        assert cls_ctx == reg_ctx, "context_nt mismatch between models"
        self.context_nt = cls_ctx

        # Load guide_length
        cls_gl_path = os.path.join(
            classification_model_path, 'assets.extra', 'guide_length.arg')
        reg_gl_path = os.path.join(
            regression_model_path, 'assets.extra', 'guide_length.arg')
        with open(cls_gl_path) as f:
            cls_gl = int(f.readline().strip())
        with open(reg_gl_path) as f:
            reg_gl = int(f.readline().strip())
        assert cls_gl == reg_gl, "guide_length mismatch between models"
        self.guide_length = cls_gl

        # Load thresholds
        if classification_threshold is None:
            thres_path = os.path.join(
                classification_model_path,
                'assets.extra', 'default_threshold.arg')
            with open(thres_path) as f:
                classification_threshold = float(f.readline().strip())
        self.classification_threshold = classification_threshold

        if regression_threshold is None:
            thres_path = os.path.join(
                regression_model_path,
                'assets.extra', 'default_threshold.arg')
            if os.path.isfile(thres_path):
                with open(thres_path) as f:
                    regression_threshold = float(f.readline().strip())
                regression_threshold += self.regression_shift
            else:
                regression_threshold = self.regression_shift
        self.regression_threshold = regression_threshold

        self._memoized_evaluations = {}
        self.min_activity = self.regression_lower_bound

    def _model_input_from_nt(self, pairs):
        """Create one-hot input from nucleotide sequences.

        Args:
            pairs: list of (target_with_context, guide) tuples

        Returns:
            numpy array of shape (n, seq_len, 8)
        """
        if len(pairs) == 0:
            return np.empty((0, 0, 8), dtype='f')

        l = 2 * self.context_nt + len(pairs[0][1])
        x = np.empty((len(pairs), l, 8), dtype='f')
        for i, (target_with_context, guide) in enumerate(pairs):
            assert len(target_with_context) == 2 * self.context_nt + len(guide)
            input_vec = []
            for pos in range(self.context_nt):
                v_target = _onehot(target_with_context[pos])
                v_guide = [0.0, 0.0, 0.0, 0.0]
                input_vec.append(v_target + v_guide)
            for pos in range(len(guide)):
                v_target = _onehot(
                    target_with_context[self.context_nt + pos])
                v_guide = _onehot(guide[pos])
                input_vec.append(v_target + v_guide)
            for pos in range(self.context_nt):
                v_target = _onehot(
                    target_with_context[self.context_nt + len(guide) + pos])
                v_guide = [0.0, 0.0, 0.0, 0.0]
                input_vec.append(v_target + v_guide)
            x[i] = np.array(input_vec, dtype='f')
        return x

    def _predict_from_onehot(self, model, pairs_onehot):
        """Predict activity from one-hot encoded input.

        Args:
            model: TF model
            pairs_onehot: numpy array of one-hot encoded pairs

        Returns:
            list of float predictions
        """
        if len(pairs_onehot) == 0:
            return []
        pred = model.call(pairs_onehot, training=False)
        return [p[0] for p in pred.numpy()]

    def _classify_and_decide(self, pairs_onehot):
        """Run classifier and decide active/inactive.

        Args:
            pairs_onehot: one-hot encoded pairs

        Returns:
            list of bool (True = active)
        """
        if len(pairs_onehot) == 0:
            return []
        scores = self._predict_from_onehot(
            self.classification_model, pairs_onehot)
        return [bool(p >= self.classification_threshold) for p in scores]

    def _regress(self, pairs_onehot):
        """Run regressor on active pairs.

        Args:
            pairs_onehot: one-hot encoded pairs (active only)

        Returns:
            list of regression scores (shifted to be >= 0)
        """
        if len(pairs_onehot) == 0:
            return []
        scores = self._predict_from_onehot(
            self.regression_model, pairs_onehot)
        scores = [max(self.regression_lower_bound, p + self.regression_shift)
                  for p in scores]
        return scores

    def _combine_model_results(self, classification_results,
                               regression_results):
        """Combine classifier and regressor outputs.

        Args:
            classification_results: list of bool (active?)
            regression_results: list of float (regression scores for active)

        Returns:
            list of (activity, highly_active) tuples
        """
        results = [(None, None)] * len(classification_results)
        j = 0
        for i in range(len(classification_results)):
            if classification_results[i] is False:
                results[i] = (0, False)
            else:
                activity = regression_results[j]
                highly_active = bool(
                    regression_results[j] >= self.regression_threshold)
                results[i] = (activity, highly_active)
                j += 1
        assert j == len(regression_results)
        return results

    def _run_models_and_memoize(self, start_pos, pairs):
        """Run both models and memoize results.

        Args:
            start_pos: start position (for memoization)
            pairs: list of (target_with_context, guide) tuples
        """
        for target_with_context, guide in pairs:
            if len(guide) != self.guide_length:
                raise ValueError(
                    f"Guide length must be {self.guide_length}")

        pairs_onehot = self._model_input_from_nt(pairs)
        classification_results = self._classify_and_decide(pairs_onehot)

        pairs_onehot_active = [
            po for po, active in zip(pairs_onehot, classification_results)
            if active]
        regression_results = self._regress(pairs_onehot_active)

        combined = self._combine_model_results(
            classification_results, regression_results)

        if start_pos not in self._memoized_evaluations:
            self._memoized_evaluations[start_pos] = {}
        mem = self._memoized_evaluations[start_pos]
        for pair, r in zip(pairs, combined):
            mem[pair] = r

    def compute_activity(self, start_pos, pairs, percentiles=None):
        """Compute activity for guide-target pairs.

        Args:
            start_pos: start position (for memoization)
            pairs: list of (target_with_context, guide) tuples
            percentiles: optional percentile(s) to compute

        Returns:
            list of activity values, or (activities, percentiles) tuple
        """
        if start_pos not in self._memoized_evaluations:
            self._memoized_evaluations[start_pos] = {}
        mem = self._memoized_evaluations[start_pos]
        unique_pairs = [pair for pair in set(pairs) if pair not in mem]
        if unique_pairs:
            self._run_models_and_memoize(start_pos, unique_pairs)

        mem = self._memoized_evaluations[start_pos]
        activities = [mem[pair][0] for pair in pairs]
        if percentiles:
            pct = np.percentile(activities, percentiles)
            return (activities, pct)
        return activities


class TmPredictor(BasePredictor):
    """Thermodynamic predictor (no ML).

    Activity = -|Tm - ideal_Tm|
    """

    def __init__(self, ideal_tm, conditions=None, reverse=False,
                 shared_memo=None):
        """Initialize TmPredictor.

        Args:
            ideal_tm: desired melting temperature
            conditions: conditions dict for primer3
            reverse: if True, guide is reverse complement
            shared_memo: shared memo dict
        """
        self.conditions = conditions
        self.ideal_tm = ideal_tm
        self.reverse = reverse
        self.context_nt = 0
        self.rough_max_activity = 0
        if shared_memo is not None:
            self._memoized_evaluations = shared_memo
        else:
            self._memoized_evaluations = {}
        self.min_activity = -self.ideal_tm

    def compute_activity(self, start_pos, pairs, percentiles=None):
        """Compute activity based on melting temperature.

        Args:
            start_pos: start position (for memoization)
            pairs: list of (target_with_context, guide) tuples
            percentiles: optional percentile(s)

        Returns:
            list of activity values
        """
        if start_pos not in self._memoized_evaluations:
            self._memoized_evaluations[start_pos] = {}
        mem = self._memoized_evaluations[start_pos]

        for pair in pairs:
            if pair not in mem:
                tm = calculate_melting_temp(
                    pair[0], pair[1], self.reverse, self.conditions)
                mem[pair] = -abs(tm - self.ideal_tm)

        activities = [mem[pair][0] if isinstance(mem[pair], list)
                      else mem[pair] for pair in pairs]
        if percentiles:
            pct = np.percentile(activities, percentiles)
            return (activities, pct)
        return activities


class SimpleBinaryPredictor:
    """Binary predictor based on mismatch binding (no ML).

    Activity = 1.0 if guide binds, 0 otherwise.
    """

    def __init__(self, mismatches, allow_gu_pairs=False,
                 required_flanking_seqs=(None, None)):
        """Initialize SimpleBinaryPredictor.

        Args:
            mismatches: max allowed mismatches
            allow_gu_pairs: if True, allow G-T pairs
            required_flanking_seqs: (s5, s3) required flanking sequences
        """
        self.mismatches = mismatches
        self.allow_gu_pairs = allow_gu_pairs
        self.required_flanking_seqs = required_flanking_seqs
        self.rough_max_activity = 1.0
        self.min_activity = 0

    def compute_activity(self, start_pos, gd_sequence, aln, percentiles=None):
        """Compute activity by checking binding across alignment.

        Args:
            start_pos: start position in alignment
            gd_sequence: guide sequence
            aln: Alignment object
            percentiles: optional percentile(s)

        Returns:
            numpy array of activities (1.0 or 0.0 per sequence)
        """
        from adapt_reimpl.sequence_utils import binds
        activities = np.zeros(aln.num_sequences)
        seqs_bound = aln.sequences_bound_by_oligo(
            gd_sequence, start_pos, self.mismatches, self.allow_gu_pairs,
            required_flanking_seqs=self.required_flanking_seqs)
        for seq_idx in seqs_bound:
            activities[seq_idx] = 1.0
        if percentiles:
            pct = np.percentile(activities, percentiles)
            return (activities, pct)
        return activities

    def cleanup_memoized(self, start_pos):
        """No memoization needed."""
        pass
