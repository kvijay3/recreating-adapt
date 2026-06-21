"""Tests for data_parser.py and sequence_utils.py."""

import os
import gzip
import tempfile
import unittest

import numpy as np

from adapt_reimpl.sequence_utils import (
    one_hot_encode_base,
    one_hot_encode,
    one_hot_encode_pair,
    convert_to_nt,
    reverse_complement,
    hamming_distance,
    binds,
    determine_consensus,
)
from adapt_reimpl.data_parser import parse_fasta, Cas13ActivityParser
from adapt_reimpl.thermo import gc_content, melting_temp


class TestSequenceUtils(unittest.TestCase):

    def test_one_hot_encode_base(self):
        result = one_hot_encode_base('A')
        self.assertEqual(result, [1.0, 0.0, 0.0, 0.0])
        result = one_hot_encode_base('T')
        self.assertEqual(result, [0.0, 0.0, 0.0, 1.0])
        result = one_hot_encode_base('N')
        self.assertAlmostEqual(sum(result), 1.0)
        self.assertEqual(len(result), 4)

    def test_one_hot_encode(self):
        seq = 'ACGT'
        result = one_hot_encode(seq)
        self.assertEqual(result.shape, (4, 4))
        self.assertTrue(np.allclose(result[0], [1, 0, 0, 0]))
        self.assertTrue(np.allclose(result[3], [0, 0, 0, 1]))

    def test_one_hot_encode_pair(self):
        target = 'AAAAAAAAAAAACGTAAAAAAAAAAA'  # 4+12+4 = 20... need 2*ctx+guide
        guide = 'ACGTACGTACGTACGTACGTACGTACGT'  # 28
        context_nt = 10
        # target needs to be 2*10 + 28 = 48
        target = 'A' * 10 + 'ACGT' * 7 + 'A' * 10  # 10 + 28 + 10 = 48
        result = one_hot_encode_pair(target, guide, context_nt)
        self.assertEqual(result.shape, (48, 8))

    def test_convert_to_nt(self):
        onehot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype='f')
        self.assertEqual(convert_to_nt(onehot), 'ACGT')

    def test_reverse_complement(self):
        self.assertEqual(reverse_complement('ACGT'), 'ACGT')
        self.assertEqual(reverse_complement('AAAA'), 'TTTT')
        self.assertEqual(reverse_complement('AC'), 'GT')

    def test_hamming_distance(self):
        self.assertEqual(hamming_distance('ACGT', 'ACGT'), 0)
        self.assertEqual(hamming_distance('ACGT', 'ACGA'), 1)
        self.assertEqual(hamming_distance('AAAA', 'TTTT'), 4)

    def test_binds(self):
        self.assertTrue(binds('ACGT', 'ACGT', 0))
        self.assertTrue(binds('ACGT', 'ACGA', 1))
        self.assertFalse(binds('ACGT', 'TTTT', 1))
        self.assertFalse(binds('AC-T', 'ACGT', 0))

    def test_determine_consensus(self):
        seqs = ['ACGT', 'ACGT', 'ACGA']
        consensus = determine_consensus(seqs)
        self.assertEqual(consensus, 'ACGT')


class TestFastaParsing(unittest.TestCase):

    def test_parse_fasta(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta',
                                         delete=False) as f:
            f.write('>seq1\nACGTACGT\n>seq2\nTTTTGGGG\n')
            f.flush()
            entries = parse_fasta(f.name)
        os.unlink(f.name)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0][0], 'seq1')
        self.assertEqual(entries[0][1], 'ACGTACGT')
        self.assertEqual(entries[1][1], 'TTTTGGGG')


class TestCas13ActivityParser(unittest.TestCase):

    def _create_test_tsv(self, path, n_rows=20):
        headers = ['guide_seq', 'target_at_guide', 'target_before',
                   'target_after', 'out_logk_measurement', 'type',
                   'guide_pos_nt']
        with gzip.open(path, 'wt') as f:
            f.write('\t'.join(headers) + '\n')
            for i in range(n_rows):
                guide = 'ACGT' * 7  # 28-mer
                target = 'ACGT' * 7
                before = 'TTTT' * 10  # 40 nt context
                after = 'GGGG' * 10
                activity = -2.0 if i % 3 == 0 else -5.0
                row_type = 'exp' if i % 2 == 0 else 'pos'
                pos = str(i * 10)
                f.write('\t'.join([guide, target, before, after,
                                   str(activity), row_type, pos]) + '\n')

    def test_parser_read(self):
        with tempfile.NamedTemporaryFile(suffix='.tsv.gz', delete=False) as f:
            path = f.name
        self._create_test_tsv(path)
        try:
            parser = Cas13ActivityParser(path, subset='exp-and-pos',
                                         context_nt=10)
            parser.set_activity_mode(classify_activity=True,
                                     regress_on_all=False,
                                     regress_only_on_active=False)
            parser.read()
            x_train, y_train = parser.train_set()
            self.assertEqual(x_train.ndim, 3)
            self.assertEqual(x_train.shape[1], 48)  # 2*10 + 28
            self.assertEqual(x_train.shape[2], 8)
            self.assertEqual(y_train.shape[1], 1)
        finally:
            os.unlink(path)


class TestThermo(unittest.TestCase):

    def test_gc_content(self):
        self.assertAlmostEqual(gc_content('ACGT'), 0.5)
        self.assertAlmostEqual(gc_content('AAAA'), 0.0)
        self.assertAlmostEqual(gc_content('GCGC'), 1.0)

    def test_melting_temp(self):
        tm = melting_temp('ACGTACGTACGTACGT')
        self.assertIsInstance(tm, float)
        self.assertGreater(tm, 0)


if __name__ == '__main__':
    unittest.main()
