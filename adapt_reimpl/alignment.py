"""Alignment class for working with aligned viral genome sequences.

Stores sequences in column-major order for efficient position-based access.
"""

import statistics
from math import log2, isclose
from collections import defaultdict

import numpy as np

from adapt_reimpl.sequence_utils import (
    FASTA_CODES,
    binds,
    query_target_eq,
    determine_consensus,
)


class Alignment:
    """Immutable collection of aligned sequences in column-major order.

    seqs[i] is a string giving the bases at position i across all sequences.
    """

    def __init__(self, seqs, seq_norm_weights=None):
        """Initialize alignment.

        Args:
            seqs: list of str in column-major order (seqs[i] = column i)
            seq_norm_weights: normalized weights per sequence (sum to 1)
        """
        self.seq_length = len(seqs)
        if self.seq_length == 0:
            raise Exception("Alignment cannot be empty")
        self.num_sequences = len(seqs[0])
        for s in seqs:
            assert len(s) == self.num_sequences

        self.seqs = seqs
        if seq_norm_weights is None:
            self.seq_norm_weights = [1.0 / self.num_sequences
                                     for _ in range(self.num_sequences)]
        else:
            assert isclose(sum(seq_norm_weights), 1.0)
            self.seq_norm_weights = seq_norm_weights

        self._frac_missing = None
        self._median_missing = None

    def extract_range(self, pos_start, pos_end):
        """Extract a sub-alignment over [pos_start, pos_end).

        Args:
            pos_start: start position (inclusive)
            pos_end: end position (exclusive)

        Returns:
            new Alignment over the specified range
        """
        return Alignment(
            self.seqs[pos_start:pos_end],
            seq_norm_weights=self.seq_norm_weights)

    def _compute_frac_missing(self):
        """Compute fraction of sequences with missing data at each position."""
        self._frac_missing = [0.0] * self.seq_length
        for j in range(self.seq_length):
            num_n = sum(1 for i in range(self.num_sequences)
                        if self.seqs[j][i] == 'N')
            self._frac_missing[j] = float(num_n) / self.num_sequences

    def frac_missing_at_pos(self, pos):
        """Fraction of sequences with missing data at a position.

        Args:
            pos: position in alignment

        Returns:
            fraction with 'N' at pos
        """
        if self._frac_missing is None:
            self._compute_frac_missing()
        return self._frac_missing[pos]

    def median_sequences_with_missing_data(self):
        """Median fraction of sequences with missing data across positions.

        Returns:
            median fraction
        """
        if self._median_missing is None:
            if self._frac_missing is None:
                self._compute_frac_missing()
            self._median_missing = statistics.median(self._frac_missing)
        return self._median_missing

    def seqs_with_gap(self, seqs_to_consider=None):
        """Find sequences that contain a gap.

        Args:
            seqs_to_consider: subset of sequence indices

        Returns:
            set of indices with gaps
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)
        has_gap = set()
        for j in range(self.seq_length):
            has_gap.update(
                i for i in seqs_to_consider if self.seqs[j][i] == '-')
        return has_gap

    def make_list_of_seqs(self, seqs_to_consider=None, include_idx=False,
                          remove_gaps=False):
        """Construct list of sequences from the alignment.

        Args:
            seqs_to_consider: subset of indices (None = all)
            include_idx: return (seq, idx) tuples
            remove_gaps: remove '-' from sequences

        Returns:
            list of str or list of (str, int) tuples
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)

        def seq_str(i):
            s = ''.join(self.seqs[j][i] for j in range(self.seq_length))
            if remove_gaps:
                s = s.replace('-', '')
            return s

        if include_idx:
            return [(seq_str(i), i) for i in seqs_to_consider]
        else:
            return [seq_str(i) for i in seqs_to_consider]

    def determine_consensus_sequence(self, seqs_to_consider=None):
        """Determine consensus sequence from the alignment.

        Args:
            seqs_to_consider: subset of indices (None = all)

        Returns:
            consensus string
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)

        consensus = ''
        for i in range(self.seq_length):
            counts = {'A': 0.0, 'T': 0.0, 'C': 0.0, 'G': 0.0}
            for j in seqs_to_consider:
                b = self.seqs[i][j]
                if b in counts:
                    counts[b] += self.seq_norm_weights[j]
                elif b == 'N':
                    continue
                elif b in FASTA_CODES:
                    for c in FASTA_CODES[b]:
                        counts[c] += (
                            self.seq_norm_weights[j] / len(FASTA_CODES[b]))
            counts_sorted = sorted(counts.items())
            max_base = max(counts_sorted, key=lambda x: x[1])[0]
            if counts[max_base] == 0:
                consensus += 'N'
            else:
                consensus += max_base
        return consensus

    def determine_most_common_sequences(self, seqs_to_consider=None,
                                        skip_ambiguity=False, n=1):
        """Find the n most common sequences in the alignment.

        Args:
            seqs_to_consider: subset of indices (None = all)
            skip_ambiguity: skip sequences with ambiguity codes
            n: number of sequences to return

        Returns:
            list of str (most common first)
        """
        if seqs_to_consider is None:
            seqs_to_consider = range(self.num_sequences)
        seqs_list = list(seqs_to_consider)

        allowed = {'A', 'T', 'C', 'G', '-'}
        seqs_str = self.make_list_of_seqs(seqs_to_consider=seqs_to_consider)
        seq_count = defaultdict(float)
        for j, s in enumerate(seqs_str):
            if skip_ambiguity:
                if any(b not in allowed for b in s):
                    continue
            seq_count[s] += self.seq_norm_weights[seqs_list[j]]

        if len(seq_count) == 0:
            return None

        counts_sorted = sorted(seq_count.items(),
                               key=lambda x: (-x[1], x[0]))
        return [s for s, _ in counts_sorted[:n]]

    def sequences_bound_by_oligo(self, olg_seq, olg_start, mismatches,
                                 allow_gu_pairs,
                                 required_flanking_seqs=(None, None)):
        """Determine which sequences an oligo hybridizes to.

        Args:
            olg_seq: oligo sequence
            olg_start: start position in alignment
            mismatches: max allowed mismatches
            allow_gu_pairs: if True, allow G-T pairs
            required_flanking_seqs: (s5, s3) required flanking

        Returns:
            list of sequence indices that bind
        """
        assert olg_start + len(olg_seq) <= self.seq_length
        aln_for_oligo = self.extract_range(olg_start, olg_start + len(olg_seq))
        seq_rows = aln_for_oligo.make_list_of_seqs(include_idx=True)

        # Check required flanking
        seqs_with_flanking = set(range(self.num_sequences))
        s5, s3 = required_flanking_seqs
        if s5 is not None and len(s5) > 0:
            if olg_start - len(s5) >= 0:
                flanking_5 = set()
                for i in range(self.num_sequences):
                    s = ''.join(self.seqs[j][i]
                                for j in range(olg_start - len(s5), olg_start))
                    if '-' not in s and query_target_eq(s5, s):
                        flanking_5.add(i)
                seqs_with_flanking &= flanking_5
            else:
                seqs_with_flanking = set()
        if s3 is not None and len(s3) > 0:
            end = olg_start + len(olg_seq)
            if end + len(s3) <= self.seq_length:
                flanking_3 = set()
                for i in range(self.num_sequences):
                    s = ''.join(self.seqs[j][i]
                                for j in range(end, end + len(s3)))
                    if '-' not in s and query_target_eq(s3, s):
                        flanking_3.add(i)
                seqs_with_flanking &= flanking_3
            else:
                seqs_with_flanking = set()

        binding_seqs = []
        for seq, seq_idx in seq_rows:
            if (seq_idx in seqs_with_flanking and
                    binds(olg_seq, seq, mismatches, allow_gu_pairs)):
                binding_seqs.append(seq_idx)
        return binding_seqs

    def compute_activity(self, start, olg_sequence, predictor, mutator=None):
        """Compute predicted activity between oligo and all target sequences.

        Args:
            start: start position in alignment
            olg_sequence: oligo sequence
            predictor: Predictor object
            mutator: optional Mutator (not implemented)

        Returns:
            numpy array of activities per sequence
        """
        from adapt_reimpl.predict_activity import SimpleBinaryPredictor

        oligo_length = len(olg_sequence)
        assert start + oligo_length <= self.seq_length

        if isinstance(predictor, SimpleBinaryPredictor):
            return predictor.compute_activity(
                start, olg_sequence, self)

        # Extract target sequences with context
        context_nt = predictor.context_nt
        if (start - context_nt < 0 or
                start + oligo_length + context_nt > self.seq_length):
            raise CannotConstructOligoError(
                "Context needed falls outside alignment range")

        aln_with_ctx = self.extract_range(
            start - context_nt,
            start + oligo_length + context_nt)

        # Ignore sequences with gaps in this region
        all_seqs = set(range(self.num_sequences))
        seqs_with_gap = set(aln_with_ctx.seqs_with_gap(all_seqs))
        seqs_to_consider = all_seqs - seqs_with_gap

        seq_rows = aln_with_ctx.make_list_of_seqs(
            seqs_to_consider, include_idx=True)

        activities = np.zeros(self.num_sequences)
        pairs_to_eval = []
        pairs_seq_idx = []
        for seq_with_ctx, seq_idx in seq_rows:
            pairs_to_eval.append((seq_with_ctx, olg_sequence))
            pairs_seq_idx.append(seq_idx)

        evals = predictor.compute_activity(start, pairs_to_eval)
        for activity, seq_idx in zip(evals, pairs_seq_idx):
            activities[seq_idx] = activity

        return activities

    def position_entropy(self):
        """Compute entropy at each position in the alignment.

        Returns:
            list of entropy values per position
        """
        position_entropy = []
        for i in range(self.seq_length):
            counts = {'A': 0.0, 'T': 0.0, 'C': 0.0, 'G': 0.0, '-': 0.0}
            for j in range(self.num_sequences):
                b = self.seqs[i][j]
                if b in counts:
                    counts[b] += self.seq_norm_weights[j]
                elif b in FASTA_CODES:
                    for c in FASTA_CODES[b]:
                        counts[c] += (
                            self.seq_norm_weights[j] / len(FASTA_CODES[b]))
            probs = [p for p in counts.values() if p > 0]
            position_entropy.append(sum(-p * log2(p) for p in probs))
        return position_entropy

    @staticmethod
    def from_list_of_seqs(seqs, seq_norm_weights=None):
        """Construct Alignment from row-major aligned sequences.

        Args:
            seqs: list of str (seqs[i] = i'th sequence, all same length)
            seq_norm_weights: optional weights

        Returns:
            Alignment object
        """
        num_sequences = len(seqs)
        if num_sequences == 0:
            raise Exception("Cannot construct alignment from 0 sequences")

        seq_length = len(seqs[0])
        for s in seqs:
            if len(s) != seq_length:
                raise ValueError("Sequences must be the same length")

        seqs_col = ['' for _ in range(seq_length)]
        for j in range(seq_length):
            seqs_col[j] = ''.join(seqs[i][j] for i in range(num_sequences))

        return Alignment(seqs_col, seq_norm_weights=seq_norm_weights)


class CannotConstructOligoError(Exception):
    """Raised when an oligo cannot be constructed at a position."""
    pass
