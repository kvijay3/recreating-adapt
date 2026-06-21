#!/usr/bin/env python
"""Preprocess raw plate reader fluorescence kinetics into ADAPT-format TSV.

Pipeline:
  1. Read plate reader output (time-series RFU per well)
  2. Read well map (well -> guide_seq, target_seq, guide_pos_nt, type, etc.)
  3. For each well:
     a. Extract RFU time-course
     b. Subtract background (negative control wells)
     c. Fit exponential model: F(t) = A*(1 - exp(-k*t)) + B
     d. Compute log10(k) as the activity measurement
  4. Aggregate replicates per guide-target pair
  5. Output curated TSV (median/stdev/count/measurements)
  6. Optionally resample to create .resampled.tsv.gz for training

Usage:
  python -m adapt_reimpl.preprocess_kinetics \
      --plate-reader data/plate_reader.csv \
      --well-map data/well_map.csv \
      --output data/CCF-curated/my_pairs_annotated.curated.tsv \
      --time-interval 5 \
      [--context-nt 20] \
      [--resample] \
      [--num-replicates 10] \
      [--background-wells A01,A02] \
      [--time-units minutes]

Plate reader CSV format (one of):
  - Long: columns [time, well, rfu]  (one row per timepoint per well)
  - Wide: first column is time, subsequent columns are well IDs (e.g. A01, B02)

Well map CSV format:
  columns: well, guide_seq, target_seq, guide_pos_nt, type, crrna_block
  - type: 'exp' (experiment), 'pos' (positive control), 'neg' (negative control)
  - guide_pos_nt: 0-based position of guide in target
  - crrna_block: integer block ID (optional, default 0)
"""

import argparse
import csv
import gzip
import math
import os
import statistics
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

CONTEXT_NT = 20
CRRNA_LEN = 28
ACTIVITY_THRESHOLD = -4.0


# ---------------------------------------------------------------------------
# Reading input files
# ---------------------------------------------------------------------------

def read_plate_reader(path):
    """Read plate reader CSV file.

    Supports two formats:
      - Long: columns [time, well, rfu]
      - Wide: first column is time, rest are well IDs

    Returns:
      dict {well: (times_array, rfu_array)}
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]

        # Detect format
        if 'time' in [h.lower() for h in header] and 'well' in [h.lower() for h in header] and 'rfu' in [h.lower() for h in header]:
            # Long format
            time_idx = [i for i, h in enumerate(header) if h.lower() == 'time'][0]
            well_idx = [i for i, h in enumerate(header) if h.lower() == 'well'][0]
            rfu_idx = [i for i, h in enumerate(header) if h.lower() == 'rfu'][0]

            data = defaultdict(lambda: ([], []))
            for row in reader:
                if len(row) <= max(time_idx, well_idx, rfu_idx):
                    continue
                well = row[well_idx].strip()
                t = float(row[time_idx])
                rfu = float(row[rfu_idx])
                data[well][0].append(t)
                data[well][1].append(rfu)

            result = {}
            for well, (times, rfus) in data.items():
                result[well] = (np.array(times), np.array(rfus))
            return result

        else:
            # Wide format: first column is time, rest are wells
            well_names = header[1:]
            data = {w: ([], []) for w in well_names}

            for row in reader:
                if len(row) < 2:
                    continue
                t = float(row[0])
                for i, well in enumerate(well_names):
                    val = row[i + 1].strip()
                    if val:
                        data[well][0].append(t)
                        data[well][1].append(float(val))

            result = {}
            for well, (times, rfus) in data.items():
                if len(times) > 0:
                    result[well] = (np.array(times), np.array(rfus))
            return result


def read_well_map(path):
    """Read well map CSV file.

    Expected columns: well, guide_seq, target_seq, guide_pos_nt, type, crrna_block

    Returns:
      list of dicts, one per well
    """
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize column names (strip whitespace)
            row = {k.strip().lower(): v.strip() for k, v in row.items() if k}
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Kinetic fitting
# ---------------------------------------------------------------------------

def exponential_model(t, A, k, B):
    """Exponential approach to saturation: F(t) = A*(1 - exp(-k*t)) + B.

    A = amplitude (F_max - F_0)
    k = first-order rate constant
    B = baseline (F_0)
    """
    return A * (1.0 - np.exp(-k * t)) + B


def fit_rate_constant(times, rfu, background_rfu=None):
    """Fit exponential model to RFU time-course and extract rate constant.

    Args:
        times: array of time points
        rfu: array of fluorescence values
        background_rfu: optional array of background (negative control) RFU
            at same time points to subtract

    Returns:
        log10(k) where k is the fitted rate constant, or None if fit fails
    """
    # Background subtraction
    if background_rfu is not None:
        rfu = rfu - background_rfu

    # Ensure non-negative after subtraction
    rfu = np.maximum(rfu, 0)

    # Initial guesses
    B0 = rfu[0] if len(rfu) > 0 else 0.0
    A0 = (rfu[-1] - rfu[0]) if len(rfu) > 1 else 1.0
    if A0 <= 0:
        A0 = max(rfu.max() - rfu.min(), 1.0)
    k0 = 0.01  # initial guess for rate constant

    # Time units: if times are in minutes, k will be per minute
    # Bounds: A > 0, k > 0, B >= 0
    try:
        popt, _ = curve_fit(
            exponential_model, times, rfu,
            p0=[A0, k0, B0],
            bounds=([0, 1e-6, 0], [np.inf, 1.0, np.inf]),
            maxfev=10000)
        A, k, B = popt
    except (RuntimeError, ValueError):
        # Fit failed — try linear approximation on early time points
        # Use the initial slope as a proxy for k
        if len(times) >= 3 and A0 > 0:
            # Linear fit on first ~20% of points
            n_init = max(3, len(times) // 5)
            early_t = times[:n_init]
            early_rfu = rfu[:n_init]
            if early_t[-1] > early_t[0]:
                slope = np.polyfit(early_t, early_rfu, 1)[0]
                k = slope / A0 if A0 > 0 else 0
                k = max(k, 1e-6)
            else:
                return None
        else:
            return None

    if k <= 0:
        return None

    log_k = math.log10(k)
    return log_k


def get_background_rfu(plate_data, background_wells, times):
    """Get average background RFU from negative control wells.

    Args:
        plate_data: dict {well: (times, rfu)}
        background_wells: list of well IDs to use as background
        times: target time points to interpolate to

    Returns:
        array of average background RFU at each time point, or None
    """
    if not background_wells:
        return None

    bg_curves = []
    for well in background_wells:
        well = well.strip()
        if well in plate_data:
            wt, wrfu = plate_data[well]
            # Interpolate to target time points
            if len(wt) == len(times) and np.allclose(wt, times):
                bg_curves.append(wrfu)
            else:
                bg_curves.append(np.interp(times, wt, wrfu))

    if not bg_curves:
        return None

    return np.mean(bg_curves, axis=0)


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def hamming_dist(a, b):
    """Compute Hamming distance between two strings."""
    assert len(a) == len(b)
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def reverse_complement(x):
    """Construct reverse complement of a DNA string."""
    rc = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'U': 'A',
          'a': 't', 'c': 'g', 'g': 'c', 't': 'a', 'u': 'a'}
    return ''.join(rc.get(b, 'N') for b in x[::-1])


def extract_context(target_seq, guide_seq, guide_pos, context_nt=CONTEXT_NT):
    """Extract target context before and after the guide.

    Args:
        target_seq: full target sequence
        guide_seq: guide sequence
        guide_pos: 0-based position of guide in target
        context_nt: number of flanking nucleotides to extract

    Returns:
        (target_at_guide, target_before, target_after)
    """
    guide_len = len(guide_seq)
    target_at_guide = target_seq[guide_pos:guide_pos + guide_len]
    target_before = target_seq[max(0, guide_pos - context_nt):guide_pos]
    target_after = target_seq[guide_pos + guide_len:guide_pos + guide_len + context_nt]

    # Pad with N's if at edge
    if len(target_before) < context_nt:
        missing = context_nt - len(target_before)
        target_before = 'N' * missing + target_before
    if len(target_after) < context_nt:
        missing = context_nt - len(target_after)
        target_after = target_after + 'N' * missing

    return target_at_guide, target_before, target_after


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_plate(plate_data, well_map_rows, context_nt, background_wells=None):
    """Process a single plate: fit kinetics and build guide-target records.

    Args:
        plate_data: dict {well: (times, rfu)}
        well_map_rows: list of well map dicts
        context_nt: number of flanking nucleotides
        background_wells: list of negative control well IDs

    Returns:
        list of dicts with curated fields
    """
    # Collect all unique time points
    all_times = None
    for well, (times, _) in plate_data.items():
        if all_times is None or len(times) > len(all_times):
            all_times = times

    # Get background
    bg_rfu = get_background_rfu(plate_data, background_wells, all_times)

    # Process each well
    results = []
    for row in well_map_rows:
        well = row.get('well', '').strip()
        if well not in plate_data:
            continue

        times, rfu = plate_data[well]

        # Fit kinetics
        # Interpolate to common time points if needed
        if bg_rfu is not None and len(times) != len(all_times):
            rfu = np.interp(all_times, times, rfu)
            times = all_times

        log_k = fit_rate_constant(times, rfu, background_rfu=bg_rfu)
        if log_k is None:
            continue

        # Clamp to activity threshold floor
        if log_k < ACTIVITY_THRESHOLD:
            log_k = ACTIVITY_THRESHOLD

        # Extract sequence info
        guide_seq = row.get('guide_seq', '').strip()
        target_seq = row.get('target_seq', '').strip()
        guide_pos = int(row.get('guide_pos_nt', 0))
        guide_type = row.get('type', 'exp').strip()
        block = int(float(row.get('crrna_block', 0)))

        if not guide_seq or not target_seq:
            continue

        target_at_guide, target_before, target_after = extract_context(
            target_seq, guide_seq, guide_pos, context_nt)

        hd = hamming_dist(guide_seq, target_at_guide)

        result = {
            'guide_seq': guide_seq,
            'guide_pos_nt': guide_pos,
            'target_at_guide': target_at_guide,
            'target_before': target_before,
            'target_after': target_after,
            'crrna_block': block,
            'type': guide_type,
            'guide_target_hamming_dist': hd,
            'log_k': log_k,
            'target_id': row.get('target', row.get('target_id', target_seq)),
            'crrna_id': row.get('crRNA', row.get('crrna', row.get('guide_seq', ''))),
        }
        results.append(result)

    return results


def aggregate_replicates(results):
    """Aggregate individual well measurements into guide-target pair summaries.

    Groups by (guide_seq, target_at_guide) and computes median, stdev, count,
    and the list of individual measurements.

    Args:
        results: list of per-well dicts

    Returns:
        list of aggregated dicts with out_logk_median, out_logk_stdev,
        out_logk_replicate_count, out_logk_measurements
    """
    groups = defaultdict(list)
    for r in results:
        key = (r['guide_seq'], r['target_at_guide'])
        groups[key].append(r)

    aggregated = []
    for key, group in groups.items():
        measurements = sorted([g['log_k'] for g in group])
        median = statistics.median(measurements)
        stdev = statistics.stdev(measurements) if len(measurements) > 1 else 0.0
        count = len(measurements)
        measurements_str = ','.join(str(v) for v in measurements)

        # Use first row as template for non-output fields
        template = group[0]
        row = {
            'guide_seq': template['guide_seq'],
            'guide_pos_nt': template['guide_pos_nt'],
            'target_at_guide': template['target_at_guide'],
            'target_before': template['target_before'],
            'target_after': template['target_after'],
            'crrna_block': template['crrna_block'],
            'type': template['type'],
            'guide_target_hamming_dist': template['guide_target_hamming_dist'],
            'out_logk_median': median,
            'out_logk_stdev': stdev,
            'out_logk_replicate_count': count,
            'out_logk_measurements': measurements_str,
        }
        aggregated.append(row)

    return aggregated


def write_curated_tsv(rows, out_path):
    """Write curated TSV file (one row per guide-target pair)."""
    cols = ['guide_seq', 'guide_pos_nt', 'target_at_guide', 'target_before',
            'target_after', 'crrna_block', 'type', 'guide_target_hamming_dist',
            'out_logk_median', 'out_logk_stdev', 'out_logk_replicate_count',
            'out_logk_measurements']

    with open(out_path, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for row in rows:
            f.write('\t'.join([str(row[c]) for c in cols]) + '\n')

    print(f"Wrote {len(rows)} guide-target pairs to {out_path}")


def write_resampled_tsvgz(rows, out_path, num_replicates=10, seed=1):
    """Write resampled TSV.gz file (one row per measurement).

    Samples num_replicates measurements (with replacement) per guide-target pair.
    """
    np.random.seed(seed)

    cols = ['guide_seq', 'guide_pos_nt', 'target_at_guide', 'target_before',
            'target_after', 'crrna_block', 'type', 'guide_target_hamming_dist',
            'out_logk_measurement']

    with gzip.open(out_path, 'wt') as f:
        f.write('\t'.join(cols) + '\n')
        total = 0
        for row in rows:
            measurements = [float(x) for x in row['out_logk_measurements'].split(',')]
            sampled = np.random.choice(measurements, size=num_replicates)

            for m in sampled:
                out_row = [str(row[c]) for c in cols[:-1]] + [str(m)]
                f.write('\t'.join(out_row) + '\n')
                total += 1

    print(f"Wrote {total} resampled measurements to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess plate reader kinetics into ADAPT-format TSV')
    parser.add_argument('--plate-reader', required=True,
                        help='Path to plate reader CSV file')
    parser.add_argument('--well-map', required=True,
                        help='Path to well map CSV file')
    parser.add_argument('--output', required=True,
                        help='Path to output curated TSV file')
    parser.add_argument('--time-interval', type=float, default=5,
                        help='Time interval between measurements in minutes (default: 5)')
    parser.add_argument('--context-nt', type=int, default=CONTEXT_NT,
                        help=f'Number of flanking nucleotides (default: {CONTEXT_NT})')
    parser.add_argument('--background-wells', type=str, default='',
                        help='Comma-separated list of negative control well IDs')
    parser.add_argument('--time-units', type=str, default='minutes',
                        choices=['minutes', 'seconds'],
                        help='Time units in plate reader file (default: minutes)')
    parser.add_argument('--resample', action='store_true',
                        help='Also create resampled .tsv.gz for training')
    parser.add_argument('--num-replicates', type=int, default=10,
                        help='Number of replicates to sample per pair (default: 10)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for resampling (default: 1)')
    args = parser.parse_args()

    # Read inputs
    print(f"Reading plate reader data from {args.plate_reader}...")
    plate_data = read_plate_reader(args.plate_reader)
    print(f"  Found {len(plate_data)} wells with data")

    print(f"Reading well map from {args.well_map}...")
    well_map_rows = read_well_map(args.well_map)
    print(f"  Found {len(well_map_rows)} wells in map")

    # Parse background wells
    bg_wells = [w.strip() for w in args.background_wells.split(',') if w.strip()]
    if bg_wells:
        print(f"Using background wells: {bg_wells}")
    else:
        # Auto-detect negative controls from well map
        bg_wells = [r.get('well', '').strip() for r in well_map_rows
                    if r.get('type', '').strip() == 'neg']
        if bg_wells:
            print(f"Auto-detected {len(bg_wells)} negative control wells as background")

    # Process plate
    print("Fitting kinetics...")
    results = process_plate(plate_data, well_map_rows, args.context_nt, bg_wells)
    print(f"  Successfully fit {len(results)} wells")

    # Aggregate replicates
    print("Aggregating replicates...")
    aggregated = aggregate_replicates(results)
    print(f"  {len(aggregated)} unique guide-target pairs")

    # Write curated TSV
    write_curated_tsv(aggregated, args.output)

    # Optionally write resampled
    if args.resample:
        resampled_path = args.output.replace('.tsv', '.resampled.tsv.gz')
        if not resampled_path.endswith('.tsv.gz'):
            resampled_path = args.output + '.resampled.tsv.gz'
        write_resampled_tsvgz(aggregated, resampled_path,
                              num_replicates=args.num_replicates,
                              seed=args.seed)

    print("Done!")


if __name__ == '__main__':
    main()
