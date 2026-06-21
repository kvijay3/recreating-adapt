"""Tests for guide_search.py — set cover and maximize-activity."""

import os
import tempfile
import unittest

import numpy as np

from adapt_reimpl.alignment import Alignment
from adapt_reimpl.guide_search import (
    GuideSearcherMinimizeGuides,
    GuideSearcherMaximizeActivity,
)


class TestGuideSearchMinimizeGuides(unittest.TestCase):

    def _create_simple_alignment(self):
        """Create a small synthetic alignment with 5 sequences."""
        seqs = [
            'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC',
            'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC',
            'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC',
            'TTTTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC',
            'TTTTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC',
        ]
        return Alignment.from_list_of_seqs(seqs)

    def test_set_cover_finds_valid_guides(self):
        """Verify set cover finds guides that cover the alignment."""
        aln = self._create_simple_alignment()
        searcher = GuideSearcherMinimizeGuides(
            aln, guide_length=28, mismatches=3, cover_frac=0.95)

        with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False) as f:
            out_path = f.name

        try:
            searcher.find_guides_with_sliding_window(
                window_size=60, out_fn=out_path)
            # Check output file exists and has content
            with open(out_path) as f:
                lines = f.readlines()
            self.assertGreater(len(lines), 1)  # header + at least one row
        finally:
            os.unlink(out_path)

    def test_set_cover_covers_all_sequences(self):
        """Verify that selected guides actually cover sequences."""
        aln = self._create_simple_alignment()
        searcher = GuideSearcherMinimizeGuides(
            aln, guide_length=28, mismatches=3, cover_frac=1.0)

        guides = searcher._find_oligos_in_window(0, 60)
        self.assertGreater(len(guides), 0)

        # Check that guides cover all sequences
        all_bound = set()
        for guide in guides:
            for pos in searcher._selected_positions[guide]:
                bound = aln.sequences_bound_by_oligo(
                    guide, pos, 3, False)
                all_bound.update(bound)
        self.assertEqual(len(all_bound), aln.num_sequences)


class TestGuideSearchMaximizeActivity(unittest.TestCase):

    def test_maximize_activity_selects_guides(self):
        """Verify maximize-activity selects valid guides."""
        seqs = [
            'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC',
            'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC',
            'ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC',
        ]
        aln = Alignment.from_list_of_seqs(seqs)

        # Use SimpleBinaryPredictor-style search (no ML predictor)
        from adapt_reimpl.predict_activity import SimpleBinaryPredictor
        predictor = SimpleBinaryPredictor(mismatches=3)

        searcher = GuideSearcherMaximizeActivity(
            aln, guide_length=28,
            soft_guide_constraint=2, hard_guide_constraint=3,
            penalty_strength=0.1,
            predictor=None)  # No ML predictor for synthetic test

        # Should find guides
        guides = searcher._find_oligos_in_window(0, 60)
        self.assertGreater(len(guides), 0)
        self.assertLessEqual(len(guides), 3)  # hard constraint


class TestAlignment(unittest.TestCase):

    def test_from_list_of_seqs(self):
        seqs = ['ACGT', 'ACGT', 'AGGT']
        aln = Alignment.from_list_of_seqs(seqs)
        self.assertEqual(aln.num_sequences, 3)
        self.assertEqual(aln.seq_length, 4)
        self.assertEqual(aln.seqs[0], 'AAA')
        self.assertEqual(aln.seqs[1], 'CCG')
        self.assertEqual(aln.seqs[2], 'GGG')
        self.assertEqual(aln.seqs[3], 'TTT')

    def test_extract_range(self):
        seqs = ['ACGTACGT', 'ACGTACGT']
        aln = Alignment.from_list_of_seqs(seqs)
        sub = aln.extract_range(2, 6)
        self.assertEqual(sub.seq_length, 4)
        # seqs are column-major: seqs[0] is position 2 across all sequences
        # 'ACGTACGT' positions 2,3,4,5 = G,T,A,C
        self.assertEqual(sub.seqs[0], 'GG')
        self.assertEqual(sub.seqs[1], 'TT')
        self.assertEqual(sub.seqs[2], 'AA')
        self.assertEqual(sub.seqs[3], 'CC')

    def test_determine_consensus(self):
        seqs = ['ACGT', 'ACGT', 'AGGT']
        aln = Alignment.from_list_of_seqs(seqs)
        consensus = aln.determine_consensus_sequence()
        self.assertEqual(consensus, 'ACGT')

    def test_sequences_bound_by_oligo(self):
        seqs = ['ACGTACGT', 'ACGTACGT', 'TTTTACGT']
        aln = Alignment.from_list_of_seqs(seqs)
        bound = aln.sequences_bound_by_oligo('ACGTACGT', 0, 0, False)
        self.assertIn(0, bound)
        self.assertIn(1, bound)
        self.assertNotIn(2, bound)


if __name__ == '__main__':
    unittest.main()
