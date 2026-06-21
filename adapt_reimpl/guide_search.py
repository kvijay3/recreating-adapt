"""Guide search algorithms for ADAPT.

Implements two objectives:
1. Minimize-guides: greedy set cover (Slavik's partial set cover)
2. Maximize-activity: submodular maximization with soft/hard constraints
"""

import math
import random
from collections import defaultdict

import numpy as np

from adapt_reimpl.alignment import CannotConstructOligoError
from adapt_reimpl.sequence_utils import binds, determine_consensus


class CannotAchieveDesiredCoverageError(Exception):
    """Raised when desired coverage cannot be achieved."""
    pass


class CannotFindAnyOligosError(Exception):
    """Raised when no oligos can be found."""
    pass


class NoPredictorError(Exception):
    """Raised when a predictor is needed but not set."""
    pass


class GuideSearcherMinimizeGuides:
    """Greedy set cover to minimize the number of guides.

    Universe = all sequences; each guide covers sequences it binds to
    within the mismatch threshold. Iteratively select the guide covering
    the most uncovered sequences until cover_frac is reached.
    """

    def __init__(self, aln, guide_length, mismatches, cover_frac,
                 missing_data_params=(1.0, 0.0, 1.0),
                 allow_gu_pairs=False,
                 required_flanking_seqs=(None, None),
                 predictor=None):
        """Initialize the searcher.

        Args:
            aln: Alignment object
            guide_length: length of guide
            mismatches: max allowed mismatches for binding
            cover_frac: fraction of sequences to cover (0, 1]
            missing_data_params: (a, b, c) for missing data threshold
            allow_gu_pairs: if True, allow G-T pairs
            required_flanking_seqs: (s5, s3) required flanking
            predictor: optional Predictor object
        """
        self.aln = aln
        self.guide_length = guide_length
        self.mismatches = mismatches
        self.cover_frac = cover_frac
        self.allow_gu_pairs = allow_gu_pairs
        self.required_flanking_seqs = required_flanking_seqs
        self.predictor = predictor
        self.obj_type = 'min'

        # Missing data threshold
        a, b, c = missing_data_params
        self.missing_threshold = min(
            a, max(b, c * self.aln.median_sequences_with_missing_data()))

        self._selected_positions = defaultdict(set)
        self._memo = {}

    def _construct_oligo_at_pos(self, start, seqs_to_consider):
        """Construct the best guide at a position.

        Uses consensus of remaining sequences as the guide, then checks
        which sequences it binds to.

        Args:
            start: start position in alignment
            seqs_to_consider: set of sequence indices still uncovered

        Returns:
            (guide_seq, covered_seqs, score) or None
        """
        if start + self.guide_length > self.aln.seq_length:
            return None

        # Check missing data
        for pos in range(start, start + self.guide_length):
            if self.aln.frac_missing_at_pos(pos) > self.missing_threshold:
                return None

        aln_for_oligo = self.aln.extract_range(
            start, start + self.guide_length)

        # Remove sequences with gaps
        seqs_with_gap = set(aln_for_oligo.seqs_with_gap(seqs_to_consider))
        valid_seqs = seqs_to_consider - seqs_with_gap

        if len(valid_seqs) == 0:
            return None

        # Construct consensus guide
        consensus = aln_for_oligo.determine_consensus_sequence(valid_seqs)
        if 'N' in consensus:
            # Try individual sequences as guides instead
            seq_rows = aln_for_oligo.make_list_of_seqs(valid_seqs,
                                                        include_idx=True)
            best_guide = None
            best_cover = set()
            for seq, seq_idx in seq_rows:
                if 'N' in seq or '-' in seq:
                    continue
                covered = set()
                for s, si in seq_rows:
                    if binds(seq, s, self.mismatches, self.allow_gu_pairs):
                        covered.add(si)
                if len(covered) > len(best_cover):
                    best_guide = seq
                    best_cover = covered
            if best_guide is None:
                return None
            return (best_guide, best_cover, len(best_cover))

        # Find sequences bound by consensus
        covered = set()
        seq_rows = aln_for_oligo.make_list_of_seqs(valid_seqs, include_idx=True)
        for seq, seq_idx in seq_rows:
            if binds(consensus, seq, self.mismatches, self.allow_gu_pairs):
                covered.add(seq_idx)

        # Also try individual sequences as alternative guides
        for seq, seq_idx in seq_rows:
            if 'N' in seq or '-' in seq:
                continue
            alt_covered = set()
            for s, si in seq_rows:
                if binds(seq, s, self.mismatches, self.allow_gu_pairs):
                    alt_covered.add(si)
            if len(alt_covered) > len(covered):
                consensus = seq
                covered = alt_covered

        score = len(covered)
        return (consensus, covered, score)

    def _find_optimal_oligo_in_window(self, start, end, universe):
        """Find the guide that covers the most sequences in a window.

        Args:
            start: window start (inclusive)
            end: window end (exclusive)
            universe: set of uncovered sequence indices

        Returns:
            (guide_seq, covered_seqs, position, score) or (None, set(), None, 0)
        """
        search_end = end - self.guide_length + 1
        best = (None, set(), None, 0)

        for pos in range(start, search_end):
            p = self._construct_oligo_at_pos(pos, universe)
            if p is not None:
                guide, covered, score = p
                if score > best[3]:
                    best = (guide, covered, pos, score)

        return best

    def _find_oligos_in_window(self, start, end):
        """Find minimal set of guides covering sequences in a window.

        Uses greedy set cover with partial cover (Slavik's algorithm).

        Args:
            start: window start (inclusive)
            end: window end (exclusive)

        Returns:
            set of guide sequences
        """
        universe = set(range(self.aln.num_sequences))
        total_weight = sum(self.aln.seq_norm_weights)
        target_weight = self.cover_frac * total_weight

        # Track covered weight
        covered_weight = 0.0
        oligos_in_cover = set()

        while covered_weight < target_weight:
            remaining = universe.copy()
            if len(remaining) == 0:
                break

            guide, covered, pos, score = self._find_optimal_oligo_in_window(
                start, end, remaining)

            if guide is None or len(covered) == 0:
                raise CannotAchieveDesiredCoverageError(
                    f"Cannot achieve coverage in window [{start}, {end})")

            oligos_in_cover.add(guide)
            self._selected_positions[guide].add(pos)
            universe -= covered
            for seq_idx in covered:
                covered_weight += self.aln.seq_norm_weights[seq_idx]

        return oligos_in_cover

    def find_guides_with_sliding_window(self, window_size, out_fn,
                                        window_step=1, sort=False):
        """Find guides across all sliding windows and write to TSV.

        Args:
            window_size: size of sliding window
            out_fn: output TSV filename
            window_step: step size for window
            sort: if True, sort by number of guides
        """
        guide_collections = []
        for start in range(0, self.aln.seq_length - window_size + 1,
                           window_step):
            end = start + window_size
            try:
                oligos = self._find_oligos_in_window(start, end)
                guide_collections.append((start, end, oligos))
            except (CannotAchieveDesiredCoverageError,
                    CannotFindAnyOligosError):
                continue

        if sort:
            guide_collections.sort(
                key=lambda x: (len(x[2]), -self._score_collection(x[2])))

        with open(out_fn, 'w') as outf:
            outf.write('\t'.join([
                'window-start', 'window-end', 'count', 'score',
                'total-frac-bound', 'target-sequences',
                'target-sequence-positions']) + '\n')
            for start, end, guide_seqs in guide_collections:
                count = len(guide_seqs)
                frac_bound = self._total_frac_bound(guide_seqs)
                guide_sorted = sorted(guide_seqs)
                positions = [list(self._selected_positions[g])[0]
                             for g in guide_sorted]
                outf.write('\t'.join(str(x) for x in [
                    start, end, count, count, frac_bound,
                    ' '.join(guide_sorted),
                    ' '.join(str(p) for p in positions)]) + '\n')

        # Print analysis
        if len(guide_collections) > 0:
            min_count = min(len(x[2]) for x in guide_collections)
            print(f"Windows scanned: {len(list(range(0, self.aln.seq_length - window_size + 1, window_step)))}")
            print(f"Windows with guides: {len(guide_collections)}")
            print(f"Min guides in a window: {min_count}")

    def _total_frac_bound(self, oligos):
        """Calculate total weighted fraction of sequences bound by oligos.

        Args:
            oligos: set of guide sequences

        Returns:
            weighted fraction
        """
        seqs_bound = set()
        for olg in oligos:
            for pos in self._selected_positions[olg]:
                bound = self.aln.sequences_bound_by_oligo(
                    olg, pos, self.mismatches, self.allow_gu_pairs,
                    required_flanking_seqs=self.required_flanking_seqs)
                seqs_bound.update(bound)
        return sum(self.aln.seq_norm_weights[i] for i in seqs_bound)

    def _score_collection(self, oligos):
        """Calculate redundancy score for a set of oligos.

        Args:
            oligos: set of guide sequences

        Returns:
            score in [0, 1]
        """
        if len(oligos) == 0:
            return 0.0
        total_frac = 0.0
        for olg in oligos:
            seqs_bound = set()
            for pos in self._selected_positions[olg]:
                bound = self.aln.sequences_bound_by_oligo(
                    olg, pos, self.mismatches, self.allow_gu_pairs)
                seqs_bound.update(bound)
            frac = sum(self.aln.seq_norm_weights[i] for i in seqs_bound)
            total_frac += min(frac, self.cover_frac)
        return total_frac / len(oligos)

    def obj_value(self, oligo_set):
        """Objective value = number of guides (to minimize)."""
        return float(len(oligo_set))


class GuideSearcherMaximizeActivity:
    """Maximize expected activity of guide set.

    Uses submodular maximization with soft/hard constraints on the number
    of guides. Objective: F(G) - penalty * max(0, |G| - soft_constraint)
    where F(G) = mean of max per-sequence activity across the guide set.
    """

    def __init__(self, aln, guide_length, soft_guide_constraint,
                 hard_guide_constraint, penalty_strength,
                 missing_data_params=(1.0, 0.0, 1.0),
                 mismatches=3,
                 allow_gu_pairs=False,
                 required_flanking_seqs=(None, None),
                 predictor=None):
        """Initialize the searcher.

        Args:
            aln: Alignment object
            guide_length: length of guide
            soft_guide_constraint: soft limit on number of guides
            hard_guide_constraint: hard limit on number of guides
            penalty_strength: penalty coefficient for exceeding soft constraint
            missing_data_params: (a, b, c) for missing data threshold
            mismatches: max mismatches for binding (used when no predictor)
            allow_gu_pairs: if True, allow G-T pairs
            required_flanking_seqs: (s5, s3) required flanking
            predictor: Predictor object (required for this objective)
        """
        self.aln = aln
        self.guide_length = guide_length
        self.soft_guide_constraint = soft_guide_constraint
        self.hard_guide_constraint = hard_guide_constraint
        self.penalty_strength = penalty_strength
        self.mismatches = mismatches
        self.allow_gu_pairs = allow_gu_pairs
        self.required_flanking_seqs = required_flanking_seqs
        self.predictor = predictor
        self.obj_type = 'max'

        a, b, c = missing_data_params
        self.missing_threshold = min(
            a, max(b, c * self.aln.median_sequences_with_missing_data()))

        self._selected_positions = defaultdict(set)
        self._memo = {}
        self.rough_max_activity = 4.0

    def _construct_ground_set(self, start, end):
        """Construct the ground set of candidate guides in a window.

        For each position, construct a consensus guide from the sequences.

        Args:
            start: window start
            end: window end

        Returns:
            dict {guide_seq: start_position} of candidate guides
        """
        ground_set = {}
        search_end = end - self.guide_length + 1

        for pos in range(start, search_end):
            if pos + self.guide_length > self.aln.seq_length:
                continue

            # Check missing data
            skip = False
            for p in range(pos, pos + self.guide_length):
                if self.aln.frac_missing_at_pos(p) > self.missing_threshold:
                    skip = True
                    break
            if skip:
                continue

            # Check context availability
            if self.predictor is not None:
                ctx = self.predictor.context_nt
                if pos - ctx < 0 or pos + self.guide_length + ctx > self.aln.seq_length:
                    continue

            aln_for_oligo = self.aln.extract_range(
                pos, pos + self.guide_length)
            seqs_with_gap = set(aln_for_oligo.seqs_with_gap())
            valid_seqs = set(range(self.aln.num_sequences)) - seqs_with_gap

            if len(valid_seqs) == 0:
                continue

            consensus = aln_for_oligo.determine_consensus_sequence(valid_seqs)
            if 'N' not in consensus and '-' not in consensus:
                ground_set[consensus] = pos

            # Also add most common sequences as candidates
            common_seqs = aln_for_oligo.determine_most_common_sequences(
                valid_seqs, skip_ambiguity=True, n=3)
            if common_seqs:
                for s in common_seqs:
                    s_clean = s.replace('-', '')
                    if len(s_clean) == self.guide_length and 'N' not in s_clean:
                        ground_set[s_clean] = pos

        return ground_set

    def _compute_guide_activities(self, start, guide_seq):
        """Compute activity of a guide against all sequences.

        Args:
            start: start position in alignment
            guide_seq: guide sequence

        Returns:
            numpy array of activities per sequence
        """
        if self.predictor is None:
            # No predictor: use binding-based activity (1.0 if binds, 0.0)
            from adapt_reimpl.sequence_utils import binds
            activities = np.zeros(self.aln.num_sequences)
            bound = self.aln.sequences_bound_by_oligo(
                guide_seq, start, self.mismatches if hasattr(self, 'mismatches') else 3,
                self.allow_gu_pairs)
            for idx in bound:
                activities[idx] = 1.0
            return activities

        try:
            return self.aln.compute_activity(start, guide_seq,
                                             self.predictor)
        except CannotConstructOligoError:
            return np.zeros(self.aln.num_sequences)

    def _find_oligos_in_window(self, start, end):
        """Find guide set maximizing expected activity in a window.

        Uses greedy submodular maximization.

        Args:
            start: window start
            end: window end

        Returns:
            set of guide sequences
        """
        ground_set = self._construct_ground_set(start, end)
        if len(ground_set) == 0:
            raise CannotFindAnyOligosError(
                f"No guides found in window [{start}, {end})")

        # Compute activities for each candidate guide
        guide_activities = {}
        for guide_seq, pos in ground_set.items():
            activities = self._compute_guide_activities(pos, guide_seq)
            guide_activities[guide_seq] = (pos, activities)

        # Greedy selection
        selected = set()
        current_max = np.zeros(self.aln.num_sequences)

        while len(selected) < self.hard_guide_constraint:
            best_guide = None
            best_delta = -float('inf')

            for guide_seq, (pos, activities) in guide_activities.items():
                if guide_seq in selected:
                    continue

                # Marginal gain in expected activity
                new_max = np.maximum(current_max, activities)
                delta_activity = np.mean(new_max) - np.mean(current_max)

                # Penalty for exceeding soft constraint
                penalty = 0.0
                if len(selected) + 1 > self.soft_guide_constraint:
                    penalty = self.penalty_strength

                delta = delta_activity - penalty

                if delta > best_delta:
                    best_delta = delta
                    best_guide = guide_seq

            if best_guide is None or best_delta <= 0:
                break

            selected.add(best_guide)
            pos, activities = guide_activities[best_guide]
            self._selected_positions[best_guide].add(pos)
            current_max = np.maximum(current_max, activities)

        return selected

    def obj_value(self, start, end, oligo_set, activities=None):
        """Compute objective value for a guide set.

        F(G) - penalty * max(0, |G| - soft_constraint)

        Args:
            start: window start
            end: window end
            oligo_set: set of guide sequences
            activities: precomputed activities (optional)

        Returns:
            objective value
        """
        if activities is None:
            activities = self.oligo_set_activities(start, end, oligo_set)
        expected_activity = np.mean(activities)
        penalty = self.penalty_strength * max(
            0, len(oligo_set) - self.soft_guide_constraint)
        return expected_activity - penalty

    def oligo_set_activities(self, start, end, oligo_set):
        """Compute max activity across guide set for each sequence.

        Args:
            start: window start
            end: window end
            oligo_set: set of guide sequences

        Returns:
            numpy array of max activities per sequence
        """
        activities = np.zeros(self.aln.num_sequences)
        for seq in oligo_set:
            if seq not in self._selected_positions:
                continue
            for pos in self._selected_positions[seq]:
                if pos < start or pos > end - len(seq):
                    continue
                olg_act = self._compute_guide_activities(pos, seq)
                activities = np.maximum(activities, olg_act)
        return activities

    def find_guides_with_sliding_window(self, window_size, out_fn,
                                        window_step=1, sort=False):
        """Find guides across all sliding windows and write to TSV.

        Args:
            window_size: size of sliding window
            out_fn: output TSV filename
            window_step: step size for window
            sort: if True, sort by objective value
        """
        guide_collections = []
        for start in range(0, self.aln.seq_length - window_size + 1,
                           window_step):
            end = start + window_size
            try:
                oligos = self._find_oligos_in_window(start, end)
                guide_collections.append((start, end, oligos))
            except (CannotFindAnyOligosError,
                    CannotAchieveDesiredCoverageError):
                continue

        if sort:
            guide_collections.sort(
                key=lambda x: self.obj_value(x[0], x[1], x[2]),
                reverse=True)

        with open(out_fn, 'w') as outf:
            outf.write('\t'.join([
                'window-start', 'window-end', 'count', 'objective-value',
                'total-frac-bound', 'guide-set-expected-activity',
                'target-sequences', 'target-sequence-positions']) + '\n')
            for start, end, guide_seqs in guide_collections:
                activities = self.oligo_set_activities(start, end, guide_seqs)
                obj = self.obj_value(start, end, guide_seqs,
                                     activities=activities)
                expected = np.mean(activities)
                guide_sorted = sorted(guide_seqs)
                positions = [list(self._selected_positions[g])[0]
                             for g in guide_sorted]
                outf.write('\t'.join(str(x) for x in [
                    start, end, len(guide_seqs), obj, 0.0, expected,
                    ' '.join(guide_sorted),
                    ' '.join(str(p) for p in positions)]) + '\n')

        if len(guide_collections) > 0:
            best_obj = max(
                self.obj_value(x[0], x[1], x[2]) for x in guide_collections)
            print(f"Windows with guides: {len(guide_collections)}")
            print(f"Best objective value: {best_obj:.4f}")
