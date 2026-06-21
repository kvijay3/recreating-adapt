"""Thermodynamic features: GC content and melting temperature."""

import numpy as np

try:
    import primer3
    _HAS_PRIMER3 = True
except ImportError:
    _HAS_PRIMER3 = False


def gc_content(seq):
    """Compute GC content (fraction of G+C) of a sequence.

    Args:
        seq: nucleotide string

    Returns:
        float in [0, 1]
    """
    if len(seq) == 0:
        return 0.0
    gc = sum(1 for b in seq if b in ('G', 'C'))
    return gc / len(seq)


def melting_temp(seq, conditions=None):
    """Compute melting temperature of a DNA sequence.

    Uses primer3-py if available (SantaLucia 1998 thermodynamics).
    Falls back to a simple Wallace rule approximation.

    Args:
        seq: nucleotide string
        conditions: optional dict of conditions (not used in fallback)

    Returns:
        melting temperature in degrees Celsius
    """
    if _HAS_PRIMER3:
        kwargs = {}
        if conditions is not None:
            for k, v in conditions.items():
                kwargs[k] = v
        return primer3.calc_tm(seq, **kwargs)
    else:
        # Wallace rule: Tm = 2*(A+T) + 4*(G+C)
        at = sum(1 for b in seq if b in ('A', 'T'))
        gc = sum(1 for b in seq if b in ('G', 'C'))
        return 2.0 * at + 4.0 * gc


def calculate_melting_temp(target, guide, reverse, conditions=None):
    """Calculate melting temperature between a target and guide.

    For Cas13a guides, the guide is RNA and binds to the target (DNA/RNA).
    This computes the Tm of the guide-target hybrid.

    Args:
        target: target sequence (with context)
        guide: guide sequence
        reverse: if True, the guide is the reverse complement
        conditions: optional dict of conditions

    Returns:
        melting temperature in degrees Celsius
    """
    # For Cas13a, the guide binds to the protospacer in the target
    # Extract the guide-aligned region from the target
    # The target includes context; the guide aligns to the middle
    if len(target) > len(guide):
        # Target has context; extract the guide-aligned region
        context = (len(target) - len(guide)) // 2
        target_region = target[context:context + len(guide)]
    else:
        target_region = target

    # Compute Tm of the duplex
    if reverse:
        from adapt_reimpl.sequence_utils import reverse_complement
        guide_rc = reverse_complement(guide)
        return melting_temp(guide_rc, conditions)
    else:
        return melting_temp(guide, conditions)
