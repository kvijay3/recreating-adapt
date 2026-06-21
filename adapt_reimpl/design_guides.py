"""Main CLI entry point for BADGERS guide design.

Supports both evolutionary and WGAN-AM exploration modes with
multi-target (mult) and variant-identification (diff) objectives.
"""

import argparse
import os
import sys

import numpy as np

from adapt_reimpl.data_parser import parse_fasta
from adapt_reimpl.predict_activity import Predictor
from adapt_reimpl.badgers_model import Cas13Mult, Cas13Diff
from adapt_reimpl.evolutionary_explorer import EvolutionaryExplorer
from adapt_reimpl.wgan_am_explorer import WGANAMExplorer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BADGERS guide design for Cas13a diagnostics")
    parser.add_argument('objective', choices=['mult', 'diff'],
                        help="Objective: 'mult' for multi-target detection, "
                             "'diff' for variant identification")
    parser.add_argument('explorer', choices=['evolutionary', 'wgan-am'],
                        help="Explorer type")
    parser.add_argument('targets', help="Path to target FASTA file")
    parser.add_argument('output', help="Output directory for results")
    parser.add_argument('--off-targets', help="Path to off-target FASTA (for 'diff')")
    parser.add_argument('--model', required=True,
                        help="Path to model directory (with classification/ "
                             "and regression/ subdirectories)")
    parser.add_argument('--guide-length', type=int, default=28)
    parser.add_argument('--context-nt', type=int, default=10)
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--num-generations', type=int, default=100)
    parser.add_argument('--mutation-rate', type=float, default=0.01)
    parser.add_argument('--num-rounds', type=int, default=10,
                        help="Number of WGAN-AM rounds")
    parser.add_argument('--num-candidates', type=int, default=200,
                        help="Number of WGAN-AM candidates per round")
    parser.add_argument('--max-mismatches', type=int, default=3,
                        help="Max artificial mismatches")
    parser.add_argument('--wgan-weights', help="Path to pre-trained WGAN weights")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top-n', type=int, default=10,
                        help="Number of top guides to output")
    return parser.parse_args()


def load_targets(fasta_path, guide_length, context_nt):
    """Load target sequences from FASTA, extracting guide-length regions.

    Each target sequence should be at least 2*context_nt + guide_length long.
    If sequences are longer, the middle region is extracted.

    Args:
        fasta_path: path to FASTA file
        guide_length: length of guide
        context_nt: context nucleotides on each side

    Returns:
        list of target sequences (with context)
    """
    entries = parse_fasta(fasta_path)
    targets = []
    needed_len = 2 * context_nt + guide_length

    for header, seq in entries:
        seq = seq.upper().replace('-', '')
        if len(seq) < needed_len:
            # Pad with N if too short
            seq = seq + 'N' * (needed_len - len(seq))
        elif len(seq) > needed_len:
            # Extract middle region
            start = (len(seq) - needed_len) // 2
            seq = seq[start:start + needed_len]
        targets.append(seq)

    return targets


def main():
    """Main entry point."""
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load predictor
    cls_path = os.path.join(args.model, 'classification')
    reg_path = os.path.join(args.model, 'regression')
    predictor = Predictor(cls_path, reg_path)

    # Load targets
    targets = load_targets(args.targets, args.guide_length, args.context_nt)
    print(f"Loaded {len(targets)} target sequences")

    # Build fitness model
    if args.objective == 'mult':
        fitness_model = Cas13Mult(
            targets, predictor,
            guide_length=args.guide_length,
            context_nt=args.context_nt)
    else:  # diff
        off_targets = []
        if args.off_targets:
            off_targets = load_targets(
                args.off_targets, args.guide_length, args.context_nt)
        fitness_model = Cas13Diff(
            targets, off_targets, predictor,
            guide_length=args.guide_length,
            context_nt=args.context_nt)

    # Run explorer
    if args.explorer == 'evolutionary':
        explorer = EvolutionaryExplorer(
            fitness_model,
            guide_length=args.guide_length,
            population_size=args.population_size,
            mutation_rate=args.mutation_rate,
            num_generations=args.num_generations,
            seed=args.seed)
        best_guide, best_fitness = explorer.run()
    else:  # wgan-am
        explorer = WGANAMExplorer(
            fitness_model,
            guide_length=args.guide_length,
            num_candidates=args.num_candidates,
            max_mismatches=args.max_mismatches,
            seed=args.seed)
        if args.wgan_weights:
            explorer.load_wgan(args.wgan_weights)
        best_guide, best_fitness = explorer.run(num_rounds=args.num_rounds)

    # Output results
    print(f"\nBest guide: {best_guide}")
    print(f"Best fitness: {best_fitness:.4f}")

    output_file = os.path.join(args.output, 'designed_guides.tsv')
    with open(output_file, 'w') as f:
        f.write('guide-sequence\tfitness\n')
        f.write(f'{best_guide}\t{best_fitness:.6f}\n')

    print(f"Results written to {output_file}")


if __name__ == '__main__':
    main()
