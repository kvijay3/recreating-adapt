"""Sequence manipulation, consensus, mismatch utilities, and one-hot encoding."""

import numpy as np


FASTA_CODES = {
    'A': {'A'},
    'T': {'T'},
    'C': {'C'},
    'G': {'G'},
    'K': {'G', 'T'},
    'M': {'A', 'C'},
    'R': {'A', 'G'},
    'Y': {'C', 'T'},
    'S': {'C', 'G'},
    'W': {'A', 'T'},
    'B': {'C', 'G', 'T'},
    'V': {'A', 'C', 'G'},
    'H': {'A', 'C', 'T'},
    'D': {'A', 'G', 'T'},
    'N': {'A', 'T', 'C', 'G'},
}

ONEHOT_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
ONEHOT_ORDER = ('A', 'C', 'G', 'T')


def one_hot_encode_base(b):
    """One-hot encode a single base (supports IUPAC ambiguity codes).

    Args:
        b: nucleotide character (A, C, G, T, or IUPAC ambiguity code)

    Returns:
        list of 4 floats; for ambiguity codes, distributes 1/n across possible bases
    """
    real_bases = FASTA_CODES.get(b, {b})
    v = [0.0, 0.0, 0.0, 0.0]
    for b_real in real_bases:
        if b_real in ONEHOT_IDX:
            v[ONEHOT_IDX[b_real]] = 1.0 / len(real_bases)
    return v


def one_hot_encode(seq):
    """One-hot encode a nucleotide sequence.

    Args:
        seq: nucleotide string (A, C, G, T, or IUPAC codes)

    Returns:
        numpy array of shape (len(seq), 4)
    """
    return np.array([one_hot_encode_base(b) for b in seq], dtype='f')


def one_hot_encode_pair(target_with_context, guide, context_nt):
    """One-hot encode a guide-target pair into 8-channel format.

    Produces shape (2*context_nt + guide_len, 8) where first 4 channels
    are target and next 4 are guide.

    Args:
        target_with_context: target sequence with context_nt flanking on each side
        guide: guide sequence
        context_nt: number of flanking nucleotides

    Returns:
        numpy array of shape (2*context_nt + len(guide), 8)
    """
    guide_len = len(guide)
    assert len(target_with_context) == 2 * context_nt + guide_len

    input_vec = []
    for pos in range(context_nt):
        v_target = one_hot_encode_base(target_with_context[pos])
        v_guide = [0.0, 0.0, 0.0, 0.0]
        input_vec.append(v_target + v_guide)
    for pos in range(guide_len):
        v_target = one_hot_encode_base(target_with_context[context_nt + pos])
        v_guide = one_hot_encode_base(guide[pos])
        input_vec.append(v_target + v_guide)
    for pos in range(context_nt):
        v_target = one_hot_encode_base(
            target_with_context[context_nt + guide_len + pos])
        v_guide = [0.0, 0.0, 0.0, 0.0]
        input_vec.append(v_target + v_guide)

    return np.array(input_vec, dtype='f')


def convert_to_nt(onehot_array):
    """Convert one-hot encoded array back to nucleotide string (argmax decode).

    Args:
        onehot_array: numpy array of shape (length, 4) or (length, 8)

    Returns:
        nucleotide string
    """
    _idx_to_nt = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    if onehot_array.ndim == 2 and onehot_array.shape[1] == 8:
        # 8-channel: decode just the target (first 4 channels)
        onehot_array = onehot_array[:, :4]

    seq = []
    for i in range(onehot_array.shape[0]):
        xi = onehot_array[i]
        if np.isclose(np.sum(xi), 0.0):
            seq.append('-')
        else:
            seq.append(_idx_to_nt[np.argmax(xi)])
    return ''.join(seq)


def reverse_complement(seq):
    """Return the reverse complement of a nucleotide sequence.

    Args:
        seq: nucleotide string

    Returns:
        reverse complement string
    """
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
            'N': 'N', '-': '-',
            'K': 'M', 'M': 'K', 'R': 'Y', 'Y': 'R',
            'S': 'S', 'W': 'W', 'B': 'V', 'V': 'B',
            'H': 'D', 'D': 'H'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))


def hamming_distance(s1, s2):
    """Compute Hamming distance between two equal-length strings.

    Args:
        s1, s2: strings of equal length

    Returns:
        integer Hamming distance
    """
    assert len(s1) == len(s2)
    return sum(1 for a, b in zip(s1, s2) if a != b)


def count_mismatches(target, guide):
    """Count mismatches between target and guide (same length).

    Args:
        target: target sequence string
        guide: guide sequence string

    Returns:
        number of mismatched positions
    """
    return hamming_distance(target, guide)


def binds(olg_seq, target_seq, mismatches, allow_gu_pairs=False):
    """Determine whether an oligo binds to a target sequence.

    Args:
        olg_seq: oligo sequence
        target_seq: target sequence (same length, may contain gaps)
        mismatches: maximum allowed mismatches
        allow_gu_pairs: if True, G-U (actually G-T in DNA) pairs count as matches

    Returns:
        True if oligo binds within mismatch threshold, False otherwise
    """
    if len(olg_seq) != len(target_seq):
        return False
    mm = 0
    for a, b in zip(olg_seq, target_seq):
        if a == '-' or b == '-':
            return False
        if a == b:
            continue
        if allow_gu_pairs:
            gu_pair = (a == 'G' and b == 'T') or (a == 'T' and b == 'G')
            if gu_pair:
                continue
        # Check IUPAC ambiguity overlap
        a_bases = FASTA_CODES.get(a, {a})
        b_bases = FASTA_CODES.get(b, {b})
        if a_bases & b_bases:
            continue
        mm += 1
        if mm > mismatches:
            return False
    return True


def query_target_eq(query, target):
    """Check if query matches target, respecting IUPAC ambiguity codes.

    Args:
        query: query sequence (may contain IUPAC codes)
        target: target sequence (may contain IUPAC codes)

    Returns:
        True if query and target overlap at every position
    """
    if len(query) != len(target):
        return False
    for q, t in zip(query, target):
        if q == '-' or t == '-':
            return False
        q_bases = FASTA_CODES.get(q, {q})
        t_bases = FASTA_CODES.get(t, {t})
        if not (q_bases & t_bases):
            return False
    return True


def determine_consensus(seqs, weights=None):
    """Determine consensus sequence from a list of aligned sequences.

    At each position, the consensus is the most common base (ignoring N).
    Ties are broken deterministically by alphabetical order.

    Args:
        seqs: list of aligned sequences (all same length)
        weights: optional list of weights for each sequence

    Returns:
        consensus string
    """
    if len(seqs) == 0:
        return ''
    seq_length = len(seqs[0])
    if weights is None:
        weights = [1.0 / len(seqs)] * len(seqs)

    consensus = ''
    for pos in range(seq_length):
        counts = {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}
        for i, seq in enumerate(seqs):
            b = seq[pos]
            if b in counts:
                counts[b] += weights[i]
            elif b == 'N':
                continue
            elif b in FASTA_CODES:
                for c in FASTA_CODES[b]:
                    counts[c] += weights[i] / len(FASTA_CODES[b])
        counts_sorted = sorted(counts.items())
        max_base = max(counts_sorted, key=lambda x: x[1])[0]
        if counts[max_base] == 0:
            consensus += 'N'
        else:
            consensus += max_base
    return consensus


def extract_cas13a_spacers(seq, guide_length=28, pfs_pattern=None):
    """Extract all valid Cas13a spacers from a sequence.

    LwaCas13a PFS: prefer 3' H (A/C/T) on the protospacer.
    This is configurable via pfs_pattern.

    Args:
        seq: target sequence string
        guide_length: length of spacer (default 28)
        pfs_pattern: set of valid PFS bases; default {'A', 'C', 'T'}

    Returns:
        list of (start_pos, spacer_seq) tuples
    """
    if pfs_pattern is None:
        pfs_pattern = {'A', 'C', 'T'}

    spacers = []
    for i in range(len(seq) - guide_length):
        spacer = seq[i:i + guide_length]
        # Check PFS: base immediately 3' of the protospacer
        pfs_pos = i + guide_length
        if pfs_pos < len(seq):
            pfs_base = seq[pfs_pos]
            pfs_bases = FASTA_CODES.get(pfs_base, {pfs_base})
            if pfs_bases & pfs_pattern:
                spacers.append((i, spacer))
        else:
            # At the end, no PFS available — still include
            spacers.append((i, spacer))
    return spacers
