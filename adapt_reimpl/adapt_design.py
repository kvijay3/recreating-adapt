"""ADAPT design CLI — sliding window and complete-targets modes.

Orchestrates guide search across viral genome alignments.
"""

import argparse
import os
import sys

from adapt_reimpl.data_parser import parse_alignment_fasta
from adapt_reimpl.alignment import Alignment
from adapt_reimpl.guide_search import (
    GuideSearcherMinimizeGuides,
    GuideSearcherMaximizeActivity,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ADAPT guide design for Cas13a diagnostics")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Sliding window mode
    sw = subparsers.add_parser('sliding-window',
                               help="Sliding window guide search")
    sw.add_argument('fasta', help="Path to aligned FASTA file")
    sw.add_argument('output', help="Output TSV file")
    sw.add_argument('--obj', choices=['minimize-guides', 'maximize-activity'],
                    default='minimize-guides', help="Objective")
    sw.add_argument('-gl', '--guide-length', type=int, default=28,
                    help="Guide length")
    sw.add_argument('-gm', '--mismatches', type=int, default=3,
                    help="Max mismatches for binding")
    sw.add_argument('-gp', '--cover-frac', type=float, default=0.95,
                    help="Coverage fraction")
    sw.add_argument('-w', '--window-size', type=int, default=200,
                    help="Window size")
    sw.add_argument('--window-step', type=int, default=1,
                    help="Window step size")
    sw.add_argument('--soft-guide-constraint', type=int, default=3,
                    help="Soft constraint on number of guides")
    sw.add_argument('--hard-guide-constraint', type=int, default=5,
                    help="Hard constraint on number of guides")
    sw.add_argument('--penalty-strength', type=float, default=0.5,
                    help="Penalty strength for exceeding soft constraint")
    sw.add_argument('--predict-cas13a-activity-model',
                    help="Path to model directory (with classification/ and "
                         "regression/ subdirectories)")
    sw.add_argument('--classification-threshold', type=float,
                    help="Classification threshold (default: from model)")
    sw.add_argument('--sort', action='store_true',
                    help="Sort output by objective")
    sw.add_argument('--allow-gu-pairs', action='store_true',
                    help="Allow G-U pairs")

    # Complete targets mode
    ct = subparsers.add_parser('complete-targets',
                               help="Complete target design (primers + guides)")
    ct.add_argument('fasta', help="Path to aligned FASTA file")
    ct.add_argument('output', help="Output TSV file")
    ct.add_argument('--obj', choices=['minimize-guides', 'maximize-activity'],
                    default='minimize-guides')
    ct.add_argument('-gl', '--guide-length', type=int, default=28)
    ct.add_argument('-gm', '--mismatches', type=int, default=3)
    ct.add_argument('-gp', '--cover-frac', type=float, default=0.95)
    ct.add_argument('--predict-cas13a-activity-model',
                    help="Path to model directory")

    return parser.parse_args()


def load_predictor(model_path, classification_threshold=None):
    """Load the CNN predictor from saved models.

    Args:
        model_path: path to directory with classification/ and regression/
        classification_threshold: optional threshold override

    Returns:
        Predictor object
    """
    from adapt_reimpl.predict_activity import Predictor

    cls_path = os.path.join(model_path, 'classification')
    reg_path = os.path.join(model_path, 'regression')
    return Predictor(cls_path, reg_path,
                     classification_threshold=classification_threshold)


def run_sliding_window(args):
    """Run sliding window guide search.

    Args:
        args: parsed arguments
    """
    # Parse alignment
    seqs = parse_alignment_fasta(args.fasta)
    aln = Alignment.from_list_of_seqs(seqs)
    print(f"Loaded alignment: {aln.num_sequences} sequences, "
          f"{aln.seq_length} positions")

    # Load predictor if specified
    predictor = None
    if args.predict_cas13a_activity_model:
        predictor = load_predictor(
            args.predict_cas13a_activity_model,
            classification_threshold=args.classification_threshold)

    # Create searcher
    missing_data_params = (1.0, 0.0, 1.0)
    if args.obj == 'minimize-guides':
        searcher = GuideSearcherMinimizeGuides(
            aln, args.guide_length, args.mismatches, args.cover_frac,
            missing_data_params=missing_data_params,
            allow_gu_pairs=args.allow_gu_pairs,
            predictor=predictor)
    else:
        searcher = GuideSearcherMaximizeActivity(
            aln, args.guide_length,
            args.soft_guide_constraint, args.hard_guide_constraint,
            args.penalty_strength,
            missing_data_params=missing_data_params,
            allow_gu_pairs=args.allow_gu_pairs,
            predictor=predictor)

    # Run search
    searcher.find_guides_with_sliding_window(
        args.window_size, args.output,
        window_step=args.window_step, sort=args.sort)
    print(f"Results written to {args.output}")


def run_complete_targets(args):
    """Run complete target design (simplified — just guides for now).

    Args:
        args: parsed arguments
    """
    # For now, delegate to sliding window with a large window
    seqs = parse_alignment_fasta(args.fasta)
    aln = Alignment.from_list_of_seqs(seqs)
    print(f"Loaded alignment: {aln.num_sequences} sequences, "
          f"{aln.seq_length} positions")

    predictor = None
    if args.predict_cas13a_activity_model:
        predictor = load_predictor(args.predict_cas13a_activity_model)

    missing_data_params = (1.0, 0.0, 1.0)
    if args.obj == 'minimize-guides':
        searcher = GuideSearcherMinimizeGuides(
            aln, args.guide_length, args.mismatches, args.cover_frac,
            missing_data_params=missing_data_params,
            predictor=predictor)
    else:
        searcher = GuideSearcherMaximizeActivity(
            aln, args.guide_length,
            3, 5, 0.5,
            missing_data_params=missing_data_params,
            predictor=predictor)

    # Use the whole alignment as one window
    searcher.find_guides_with_sliding_window(
        aln.seq_length, args.output, window_step=1)
    print(f"Results written to {args.output}")


def main():
    """Main entry point."""
    args = parse_args()
    if args.mode == 'sliding-window':
        run_sliding_window(args)
    elif args.mode == 'complete-targets':
        run_complete_targets(args)


if __name__ == '__main__':
    main()
