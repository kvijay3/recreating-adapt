# ADAPT + TIA/TRACE — Two-Algorithm Plan

**Date:** 2026-09-25
**Goal:** Extend ADAPT so that crRNA design accounts for **TIA/TRACE** (guide "badness" signal), not just on-target activity. Two candidate approaches are planned below:

1. **Algorithm 1 — Loop / post-hoc scoring** (non-invasive: keep ADAPT, bolt on a good-vs-bad classifier that annotates ADAPT's output).
2. **Algorithm 2 — Full retrain** (invasive: retrain ADAPT's activity model with TIA/TRACE-augmented data so high-TIA guides are omitted at design time).

---

## 0. Terminology & assumptions ⚠️ (please confirm)

These definitions drive the whole plan — flagging them because they aren't defined anywhere in the repo yet:

- **TIA** — assumed to mean **Target-Independent Activity**: collateral/background Cas13a cleavage that fires *without* (or regardless of) the intended target. High TIA ⇒ false positives / high background ⇒ **bad** for a diagnostic.
- **TRACE** — assumed to be the **assay / readout** that produces the TIA measurement (an experimental protocol name). Treated here as "the data source that labels a guide as good vs bad."
- **Key gap being addressed:** ADAPT's current model predicts only **on-target activity** = `log10(k)` (verified against the paper — see repo root discussion). It has **no notion of background/TIA**, so it can happily rank a high-background guide as "great."

> If TIA/TRACE mean something different in your context (e.g., a specific numeric metric, a different assay), tell me and I'll adjust — the algorithm shapes stay the same, only the label/target definition changes.

---

## 1. Shared prerequisite: the TIA/TRACE dataset

Neither algorithm can proceed without labeled TIA/TRACE data. Before either:

- **Need:** per-guide (and ideally per guide–target) TIA measurements from TRACE, with enough coverage to train/validate.
- **Label form to decide:**
  - Binary: `good` (low TIA) vs `bad` (high TIA) — simplest, feeds Algorithm 1 directly.
  - Continuous: a TIA score/rate — enables regression and a tunable threshold (better for Algorithm 2).
- **Join key:** how do TRACE guides map onto ADAPT's guide representation? (spacer sequence, guide position, target context). This mapping must be exact or scores won't align.
- **Overlap check:** how much do TRACE-tested guides overlap the 19,209-pair on-target training set? Overlap enables a combined model; disjoint sets force a two-model design.

**Action item:** point me at (or generate) the TIA/TRACE table before implementation starts.

---

## 2. Algorithm 1 — Loop / post-hoc TIA scoring

**Idea:** leave ADAPT untouched. Run it, then score each designed guide with a *separate* good-vs-bad classifier and **append a new column**. Optionally close the loop: use the TIA score to re-rank/filter and re-run ADAPT until the guide set is both high on-target and low TIA.

### Data flow
```
input seqs ──► ADAPT design ──► candidate guides + on-target activity
                                        │
                                        ▼
                          TIA classifier (good/bad, p_good)
                                        │
                                        ▼
                    ADAPT output + NEW COLUMN(s): tia_score, tia_class
                                        │
              ┌─────────── loop (optional) ──────────┐
              ▼                                       │
   filter / penalize high-TIA guides ──► re-run ADAPT with those excluded
```

### Components
- **Model:** "classifier of good vs bad TRACE/TIA." Start with the *best on-target model architecture we already have* (the CNN encoder in `model.py`) re-headed for TIA, or a lighter gradient-boosted model (XGBoost/LightGBM) on engineered features if TIA data is small.
- **Features (candidates):** same one-hot guide+context encoding ADAPT uses; plus GC content, homopolymer runs, predicted secondary structure / self-complementarity (`thermo.py` already has some thermodynamics), PFS allele.
- **Output columns added to ADAPT's table:** `p_good` (probability), `tia_class` (good/bad at a chosen threshold), optionally `tia_score` (continuous).
- **Loop policy (optional):** re-rank by a combined score `on_target − λ·TIA_risk`, drop guides above a TIA threshold, re-run set-cover so coverage is preserved with clean guides.

### Files touched (new work is additive)
- New: `adapt_reimpl/tia_classifier.py` (train + predict).
- New/extend: a post-processing hook that reads ADAPT output and writes the annotated table (small wrapper around `design_guides.py` / `adapt_design.py` output).
- Reuse: encoders in `data_parser.py`, CNN blocks in `model.py`, thermodynamics in `thermo.py`.
- No change to trained on-target weights.

### Pros / cons
- ✅ Fast, modular, reversible; doesn't risk ADAPT's validated on-target performance.
- ✅ Can ship the "new column" immediately; loop is an optional upgrade.
- ✅ Works even if TIA data is small or only partially overlaps the on-target set.
- ❌ TIA-awareness is *post-hoc* — search isn't TIA-guided unless you enable the loop.
- ❌ Two models to maintain.

### Milestones
1. Build TIA classifier, report AUC/PR on held-out TIA set.
2. Wire the annotation column into ADAPT output.
3. (Optional) Implement the filter-and-re-run loop; measure coverage retained vs TIA reduced.

---

## 3. Algorithm 2 — Retrain ADAPT with TIA/TRACE data

**Idea:** bake TIA into the model itself so high-TIA guides are **omitted at design time**, exactly as you described ("retrain adapt entirely … and see if we can omit the high TIA outputs").

### Sub-approaches (pick one to prototype first)
- **(a) Multi-task model** — add a second head to the CNN predicting TIA alongside on-target `log10(k)`. Design objective becomes `activity − λ·TIA`, and `guide_search.py` optimizes that directly. *Most principled; keeps on-target signal intact.*
- **(b) Relabel / filter training data** — treat high-TIA guides as inactive (or down-weight / drop them) in the training set, so the retrained hurdle model naturally scores them low. *Simplest; directly matches "omit high TIA outputs," but throws away information and can bias the on-target regressor.*
- **(c) Penalized target** — train on a combined scalar `log10(k) − λ·TIA` as the regression target. *Middle ground; single head, but conflates two signals.*

### Objective / design-time behavior
- The search (`guide_search.py`, `adapt_design.py`) already maximizes predicted activity. After retrain, "activity" incorporates a TIA penalty ⇒ high-TIA guides fall out of the maximize-activity / set-cover selection without any post-filter.

### Files touched
- `preprocess_kinetics.py` — add TIA/TRACE extraction alongside `log10(k)` (new column(s) in the curated TSV).
- `data_parser.py` — parse the TIA column; expose it as a label/target.
- `model.py` — second output head (approach a) or reuse existing head (b/c).
- `train_model.py` — multi-task loss / class-weighting / data relabeling; retrain classifier + regressor.
- `guide_search.py` / `adapt_design.py` — combined objective with tunable `λ`.

### Pros / cons
- ✅ TIA-awareness is intrinsic — cleaner one-model pipeline; best-case highest-quality guides.
- ✅ Directly answers the research question: *does training on TIA let ADAPT avoid bad guides?*
- ❌ Expensive: full retrain + re-validation against the paper baselines (must confirm on-target metrics don't regress).
- ❌ Needs enough TIA-labeled data overlapping the training pairs; risk of bias (approach b) or a hard-to-tune `λ` (a/c).
- ❌ Harder to roll back.

### Milestones
1. Extend preprocessing + parser to carry TIA.
2. Prototype approach (a) or (b) on a small split; sanity-check it doesn't tank on-target metrics.
3. Full retrain; compare guide sets vs current ADAPT (how many high-TIA guides are dropped, coverage retained).

---

## 4. Algorithm 1 vs Algorithm 2

| Dimension | Algo 1 (Loop / post-hoc) | Algo 2 (Retrain) |
|---|---|---|
| Touches ADAPT weights | No | Yes |
| TIA-aware search | Only if loop enabled | Yes, intrinsic |
| Data needed | Modest; can be disjoint from on-target set | Substantial; ideally overlapping |
| Effort / risk | Low | High |
| Reversible | Easily | Not easily |
| Time to first result | Days | Weeks |
| Best when | You want a shippable "TIA column" fast | You want TIA baked into design & to test the hypothesis |

**Recommended sequencing:** do **Algorithm 1 first** (fast signal, produces the annotated column and a reusable TIA classifier), then use its classifier and findings to inform **Algorithm 2** (the classifier's features/labels transfer directly into the multi-task retrain).

---

## 5. Evaluation (both algorithms)

- **TIA model quality:** AUC-ROC / AUC-PR (classifier) or Pearson/Spearman (regressor) on a held-out TIA set, using the same non-overlapping split discipline ADAPT already uses (position-based, to avoid leakage).
- **Design-level:** for a fixed input, compare guide sets before vs after — % high-TIA guides removed, on-target activity retained, target coverage retained.
- **On-target regression guard (Algo 2 only):** re-confirm classifier accuracy/AUC and regressor MSE/Pearson stay ≥ current README baselines.

---

## 6. Open questions

- What exactly are **TIA** and **TRACE** (metric definition, units, assay)?
- Is the TIA label **binary or continuous**?
- Where is the **TIA/TRACE dataset**, and how does it key onto ADAPT guides?
- How much does it **overlap** the 19,209-pair on-target set?
- Preferred **operating point** — hard-omit high-TIA guides, or rank with a tunable penalty `λ`?

---

## 7. Status

- [x] Folder + plan created
- [ ] TIA/TRACE definitions confirmed
- [ ] TIA/TRACE dataset located
- [ ] Algorithm 1 prototype
- [ ] Algorithm 2 prototype
