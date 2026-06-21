"""FASTA parsing, TSV training data parsing, and one-hot encoding for Cas13a."""

import gzip
import os
import random
from collections import defaultdict

import numpy as np

from adapt_reimpl.sequence_utils import (
    one_hot_encode_base,
    FASTA_CODES,
    ONEHOT_IDX,
)


def parse_fasta(path):
    """Parse a FASTA file into a list of (header, sequence) tuples.

    Args:
        path: path to FASTA file

    Returns:
        list of (header, sequence) tuples
    """
    entries = []
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if header is not None:
                    entries.append((header, ''.join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        entries.append((header, ''.join(seq_parts)))
    return entries


def parse_alignment_fasta(path):
    """Parse a FASTA alignment file into a list of aligned sequences.

    Args:
        path: path to FASTA file containing aligned sequences

    Returns:
        list of sequence strings (with gaps preserved)
    """
    entries = parse_fasta(path)
    return [seq for _, seq in entries]


class Cas13ActivityParser:
    """Parse data from paired crRNA/target Cas13 data.

    Reads the CCF TSV training data and produces one-hot encoded
    inputs with activity outputs for training CNN models.
    """

    CRRNA_LEN = 28
    SEED_START = int(CRRNA_LEN * 1 / 3)
    SEED_END = int(CRRNA_LEN * 2 / 3) + 1
    ACTIVITY_THRESHOLD = -4.0

    def __init__(self, tsv_path, subset=None, context_nt=10,
                 split=(0.8, 0.1, 0.1), shuffle_seed=1,
                 stratify_randomly=False, stratify_by_pos=False):
        """Initialize the parser.

        Args:
            tsv_path: path to the TSV (or .tsv.gz) file
            subset: None, 'exp', 'pos', 'neg', or 'exp-and-pos'
            context_nt: nt of target sequence context to include
            split: (train, validation, test) fractions
            shuffle_seed: random seed for shuffling
            stratify_randomly: if True, shuffle rows before splitting
            stratify_by_pos: if True, sort by position before splitting
        """
        assert subset in (None, 'exp', 'pos', 'neg', 'exp-and-pos')
        self.tsv_path = tsv_path
        self.subset = subset
        self.context_nt = context_nt
        assert abs(sum(split) - 1.0) < 1e-6
        self.split_train, self.split_validate, self.split_test = split
        self.stratify_randomly = stratify_randomly
        self.stratify_by_pos = stratify_by_pos
        random.seed(shuffle_seed)

        self.classify_activity = False
        self.regress_on_all = False
        self.regress_only_on_active = False
        self.was_read = False
        self._memoized_evaluations = {}

    def set_activity_mode(self, classify_activity, regress_on_all,
                          regress_only_on_active):
        """Set mode for which points to read regarding their activity.

        Args:
            classify_activity: if True, output binary labels
            regress_on_all: if True, output all pairs for regression
            regress_only_on_active: if True, only output active pairs
        """
        num_set = (int(classify_activity) + int(regress_on_all) +
                   int(regress_only_on_active))
        if num_set != 1:
            raise Exception(
                "Exactly one of classify_activity, regress_on_all, "
                "regress_only_on_active can be set")
        self.classify_activity = classify_activity
        self.regress_on_all = regress_on_all
        self.regress_only_on_active = regress_only_on_active

    def _gen_input_and_output(self, row):
        """Generate input features and output for each row.

        Produces 8-channel one-hot encoding: 4 bits target + 4 bits guide
        at each position.

        Args:
            row: dict representing row of data (keyed by column name)

        Returns:
            tuple (input_feats, output) where input_feats is a numpy array
            of shape (2*context_nt + guide_len, 8)
        """
        # Build input features
        input_feats = []

        # Context before
        context_before = row['target_before']
        assert self.context_nt <= len(context_before)
        start = len(context_before) - self.context_nt
        for pos in range(start, len(context_before)):
            v_target = one_hot_encode_base(context_before[pos])
            v_guide = [0.0, 0.0, 0.0, 0.0]
            input_feats.append(v_target + v_guide)

        # Guide region
        target = row['target_at_guide']
        guide = row['guide_seq']
        assert len(target) == len(guide)
        for pos in range(len(guide)):
            v_target = one_hot_encode_base(target[pos])
            v_guide = one_hot_encode_base(guide[pos])
            input_feats.append(v_target + v_guide)

        # Context after
        context_after = row['target_after']
        assert self.context_nt <= len(context_after)
        for pos in range(self.context_nt):
            v_target = one_hot_encode_base(context_after[pos])
            v_guide = [0.0, 0.0, 0.0, 0.0]
            input_feats.append(v_target + v_guide)

        input_feats = np.array(input_feats, dtype='f')

        # Determine output
        activity = float(row['out_logk_measurement'])
        if self.classify_activity:
            if activity <= self.ACTIVITY_THRESHOLD:
                activity = 0
            else:
                activity = 1

        return (input_feats, activity)

    def read(self):
        """Read and parse TSV file, split into train/validate/test sets."""
        header_idx = {}
        rows = []

        open_fn = gzip.open if self.tsv_path.endswith('.gz') else open
        with open_fn(self.tsv_path, 'rt') as f:
            for i, line in enumerate(f):
                ls = line.rstrip().split('\t')
                if i == 0:
                    for j in range(len(ls)):
                        header_idx[ls[j]] = j
                else:
                    rows.append(ls)

        # Convert rows to dicts
        rows = [{k: row[header_idx[k]] for k in header_idx} for row in rows]

        # Filter by subset
        if self.subset == 'exp':
            rows = [r for r in rows if r['type'] == 'exp']
        elif self.subset == 'pos':
            rows = [r for r in rows if r['type'] == 'pos']
        elif self.subset == 'neg':
            rows = [r for r in rows if r['type'] == 'neg']
        elif self.subset == 'exp-and-pos':
            rows = [r for r in rows if r['type'] in ('exp', 'pos')]

        # Shuffle or sort
        if self.stratify_randomly:
            random.shuffle(rows)
        if self.stratify_by_pos:
            rows = sorted(rows, key=lambda x: float(x['guide_pos_nt']))

        # Remove inactive points for regress_only_on_active
        if self.regress_only_on_active:
            rows = [r for r in rows
                    if float(r['out_logk_measurement']) > self.ACTIVITY_THRESHOLD]

        # Generate inputs and outputs
        inputs_and_outputs = []
        row_idx_pos = []
        self.input_feats_pos = {}
        for row in rows:
            pos = int(row['guide_pos_nt'])
            input_feats, output = self._gen_input_and_output(row)
            inputs_and_outputs.append((input_feats, output))
            row_idx_pos.append(pos)
            # Store position keyed by input features for later lookup
            input_feats_key = input_feats.tobytes()
            self.input_feats_pos[input_feats_key] = pos

        # Split into train, validate, test
        train_end_idx = int(len(inputs_and_outputs) * self.split_train)
        validate_end_idx = int(
            len(inputs_and_outputs) * (self.split_train + self.split_validate))
        self._train_set = []
        self._validate_set = []
        self._test_set = []
        for i in range(len(inputs_and_outputs)):
            if i <= train_end_idx:
                self._train_set.append(inputs_and_outputs[i])
            elif i <= validate_end_idx:
                self._validate_set.append(inputs_and_outputs[i])
            else:
                if self.stratify_by_pos:
                    last_validate_pos = row_idx_pos[validate_end_idx]
                    if row_idx_pos[i] == last_validate_pos:
                        if self.split_validate == 0:
                            self._train_set.append(inputs_and_outputs[i])
                        else:
                            self._validate_set.append(inputs_and_outputs[i])
                    else:
                        self._test_set.append(inputs_and_outputs[i])
                else:
                    self._test_set.append(inputs_and_outputs[i])

        self.was_read = True

        if self.stratify_by_pos:
            # Remove test data that overlaps with train/validate in nt space
            train_and_validate = self._train_set + self._validate_set
            self._test_set = self._make_nonoverlapping(
                train_and_validate, self._test_set)
            random.shuffle(self._train_set)
            random.shuffle(self._validate_set)
            random.shuffle(self._test_set)

    def pos_for_input(self, x):
        """Return the guide position for a given input feature array.

        Args:
            x: numpy array of input features

        Returns:
            int position
        """
        x_key = x.tobytes()
        return self.input_feats_pos[x_key]

    def _make_nonoverlapping(self, data1, data2):
        """Remove data2 points that overlap data1 in nucleotide space.

        Each data point covers a range of nucleotides (pos, pos + CRRNA_LEN).
        This removes any point in data2 whose range overlaps with any
        point in data1, preventing train/test leakage.

        Args:
            data1: list of (X, y) tuples (train+validate)
            data2: list of (X, y) tuples (test)

        Returns:
            filtered data2 with non-overlapping points
        """
        CRRNA_LEN = 28
        # Build set of all nucleotides covered by data1
        data1_ranges = set()
        for X, y in data1:
            start_pos = self.pos_for_input(X)
            data1_ranges.add((start_pos, start_pos + CRRNA_LEN))
        data1_nt = set()
        for start, end in data1_ranges:
            for p in range(start, end):
                data1_nt.add(p)

        # Filter data2
        data2_nonoverlapping = []
        for X, y in data2:
            start_pos = self.pos_for_input(X)
            X_start, X_end = start_pos, start_pos + CRRNA_LEN
            include = True
            for p in range(X_start, X_end):
                if p in data1_nt:
                    include = False
                    break
            if include:
                data2_nonoverlapping.append((X, y))
        return data2_nonoverlapping

    def _data_set(self, data):
        """Return (X, y) numpy arrays from a data set.

        Args:
            data: list of (input_feats, output) tuples

        Returns:
            (X, y) where X is shape (n, seq_len, 8) and y is shape (n, 1)
        """
        if not self.was_read:
            raise Exception("read() must be called first")
        inputs = [item[0] for item in data]
        outputs = [[item[1]] for item in data]
        return np.array(inputs, dtype='f'), np.array(outputs, dtype='f')

    def train_set(self):
        """Return training set as (X, y)."""
        return self._data_set(self._train_set)

    def validate_set(self):
        """Return validation set as (X, y)."""
        return self._data_set(self._validate_set)

    def test_set(self):
        """Return test set as (X, y)."""
        return self._data_set(self._test_set)
