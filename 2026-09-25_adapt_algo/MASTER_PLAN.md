# ADAPT + TIA/TRACE — Master Plan & Orchestration Brief

**Date:** 2026-09-25
**Audience:** an orchestrator agent that will delegate to subagents.
**This file is self-contained** — you do not need any prior chat context to execute it. Read it top to bottom, confirm the blockers in §2, then dispatch the task graph in §5.

---

## 1. Context (ground truth — do not re-derive)

**Repo:** `recreating-adapt` — a from-scratch Python reimplementation of ADAPT (and BADGERS) for designing CRISPR-Cas13a diagnostic crRNA guides. Keras 3 / TensorFlow.

**What ADAPT does today:**
- Sliding-window guide search + greedy set-cover (minimize-guides) and maximize-activity objectives.
- A two-stage **hurdle model**: a classifier (active vs inactive) + a regressor (activity of active pairs).
- Activity target = **`log10(k)`**, where `k` is the first-order Cas13a reporter-cleavage rate. This is **verified against the ADAPT paper** (Metsky et al., *Nat Biotechnol* 2022, PMC9287178):
  - Fit `F(t) = A·(1 − e^(−k·t)) + B` (A=saturation, B=background, k=rate).
  - Activity = `log(k)`; pairs with `log10(k) ≥ −4` labeled **active**, else inactive (`−4` is the floor/threshold).
  - Training set = **19,209 unique guide–target pairs** derived from **94 tiling crRNA guides** (87 experimental + 3 neg + 4 pos controls) × 229 targets.

**Key files:**
| File | Role |
|---|---|
| `adapt_reimpl/preprocess_kinetics.py` | raw plate-reader RFU → `log10(k)` curated TSV |
| `adapt_reimpl/data_parser.py` | parse curated data, one-hot encode guide+context |
| `adapt_reimpl/model.py` | CNN classifier + regressor (custom `LocallyConnected1D`) |
| `adapt_reimpl/train_model.py` | training loop, splits, class weights |
| `adapt_reimpl/predict_activity.py` | load models, predict activity |
| `adapt_reimpl/guide_search.py` | sliding-window search |
| `adapt_reimpl/adapt_design.py`, `design_guides.py` | set-cover / maximize-activity design |
| `adapt_reimpl/thermo.py` | thermodynamic features |
| `tests/` | unit tests (run these to establish a green baseline) |

**The gap this plan fixes:** ADAPT models only **on-target** activity. It has **no notion of TIA** (background / target-independent activity), so it can rank a high-background guide as excellent. Both algorithms below add a TIA-awareness signal.

---

## 2. BLOCKING prerequisites — resolve before any training task

⚠️ **Do not fabricate TIA/TRACE data or invent results.** If the items below are unmet, build and unit-test the *code paths* against the documented synthetic fixture (§4), mark every training/eval task **BLOCKED-ON-DATA**, and surface the questions to the user rather than guessing.

1. **Definitions (confirm with user):**
   - **TIA** — assumed **Target-Independent Activity**: collateral/background cleavage that fires without/regardless of the intended target. High TIA = false positives = **bad**.
   - **TRACE** — assumed to be the **assay/readout** that produces the TIA measurement.
   - If these are wrong, only the *label definition* changes; the algorithm shapes hold.
2. **Dataset:** locate the TIA/TRACE table. Needed: per-guide (ideally per guide–target) TIA values.
3. **Label form:** binary (good/bad) or continuous (TIA score)? Binary → Algo 1 directly; continuous → enables regression + tunable threshold (better for Algo 2).
4. **Join key:** exact mapping from TRACE guides to ADAPT's representation (spacer sequence, guide position, 20-nt target context). Misalignment silently corrupts everything.
5. **Overlap:** how much do TRACE guides overlap the 19,209-pair on-target set? High overlap → single combined model feasible (Algo 2a); disjoint → two-model design (Algo 1).
6. **Operating point:** hard-omit high-TIA guides, or rank with a tunable penalty `λ`?

---

## 3. The two algorithms

### Algorithm 1 — Loop / post-hoc TIA scoring (non-invasive)
Keep ADAPT's trained weights untouched. Run ADAPT, then score each designed guide with a **separate good-vs-bad classifier** and **append a new column**. Optionally close the loop.

```
input seqs ─► ADAPT design ─► candidate guides + on-target activity
                                     │
                                     ▼
                       TIA classifier (p_good, class)
                                     │
                                     ▼
              ADAPT output + NEW COLUMN(s): p_good, tia_class[, tia_score]
                                     │
        ┌──────────── loop (optional) ───────────┐
        ▼                                         │
 drop/penalize high-TIA guides ─► re-run set-cover (coverage preserved)
```
- **Model:** reuse the CNN encoder in `model.py` re-headed for TIA, **or** a lighter GBM (XGBoost/LightGBM) on engineered features if TIA data is small.
- **Features:** ADAPT's one-hot guide+context encoding; GC content; homopolymer runs; self-complementarity / secondary structure (extend `thermo.py`); PFS allele.
- **Output columns:** `p_good`, `tia_class`, optional `tia_score`.
- **Pros:** fast, modular, reversible, no risk to validated on-target performance, works with small/disjoint data. **Cons:** TIA-awareness is post-hoc unless the loop is enabled; two models to maintain.

### Algorithm 2 — Retrain ADAPT with TIA/TRACE data (invasive)
Bake TIA into the model so high-TIA guides are **omitted at design time**. Pick one sub-approach to prototype first:
- **(a) Multi-task** — add a second head predicting TIA alongside `log10(k)`; design objective = `activity − λ·TIA`; `guide_search.py` optimizes it directly. *Most principled.*
- **(b) Relabel/filter** — treat high-TIA guides as inactive (or down-weight/drop) in training so the hurdle model scores them low. *Simplest; directly matches "omit high-TIA outputs" but discards info and can bias the regressor.*
- **(c) Penalized target** — regress on `log10(k) − λ·TIA`. *Middle ground.*
- **Pros:** intrinsic TIA-awareness, single clean pipeline, directly tests the hypothesis. **Cons:** expensive retrain + re-validation; needs enough overlapping TIA data; risk of on-target regression; `λ` tuning; hard to roll back.

### Comparison
| Dimension | Algo 1 (Loop) | Algo 2 (Retrain) |
|---|---|---|
| Touches ADAPT weights | No | Yes |
| TIA-aware search | Only if loop on | Intrinsic |
| Data needed | Modest, can be disjoint | Substantial, ideally overlapping |
| Effort / risk | Low | High |
| Reversible | Easily | Not easily |
| Time to first result | Days | Weeks |

**Sequencing:** ship **Algo 1 first** (fast, produces the annotated column + a reusable TIA classifier), then transfer its labels/features into **Algo 2**.

---

## 4. Synthetic fixture (so code work can start before real data lands)

If real TIA data is unavailable, generate a small deterministic fixture matching the expected schema so subagents can build and unit-test end-to-end without fabricating scientific results. Document it as clearly synthetic.

- **TIA table schema:** `guide_seq, target_seq, guide_pos_nt, tia_value` (+ derived `tia_label` = `bad` if `tia_value ≥ threshold`).
- **Rule:** keep it obviously synthetic (e.g., `tia_value` a fixed function of GC/homopolymer content + seeded noise). Never present metrics from the fixture as real performance.

---

## 5. Subagent task graph

Each task lists: **goal · inputs · files · deliverable · done-when · deps**. Tasks in the same "Parallel group" have no interdependencies and can be dispatched concurrently.

### Phase 0 — Setup (do first)
- **T0.1 Baseline green** — *goal:* establish a working checkout. *files:* `tests/`. *deliverable:* test run log. *done-when:* `pytest` passes (or failures documented). *deps:* none.
- **T0.2 Prereq resolution** — *goal:* get §2 answered. *deliverable:* filled §2 answers or a BLOCKED note + questions to user. *done-when:* data located OR fixture (§4) chosen. *deps:* none.

### Phase 1 — Algorithm 1
*Parallel group A (after T0.2):*
- **T1.1 TIA data ingestion + split** — *inputs:* TIA table (or fixture). *files:* new `adapt_reimpl/tia_dataset.py`; reuse `data_parser.py`. *deliverable:* loader + **position-based non-overlapping** train/test split (mirror ADAPT's anti-leakage discipline). *done-when:* split reproducible, no guide-position leakage. *deps:* T0.2.
- **T1.2 Feature module** — *files:* extend `thermo.py`, reuse `data_parser.py` one-hot. *deliverable:* `extract_features(guide, context)`. *done-when:* returns encoding + engineered features with tests. *deps:* T0.2.
- **T1.4a Annotation scaffold** — *goal:* read ADAPT's output format and define where the new columns attach. *files:* new post-processing wrapper around `design_guides.py`/`adapt_design.py` output. *deliverable:* CLI stub that round-trips ADAPT output unchanged. *done-when:* passes I/O test on a sample ADAPT output. *deps:* T0.1.

*Sequential (after group A):*
- **T1.3 Train TIA classifier** — *files:* new `adapt_reimpl/tia_classifier.py`. *deliverable:* trained model + metrics (AUC-ROC, AUC-PR; or Pearson/Spearman if continuous). *done-when:* metrics reported on held-out set (or BLOCKED-ON-DATA with code tested on fixture). *deps:* T1.1, T1.2.
- **T1.4b Wire annotation column** — *deliverable:* ADAPT output + `p_good`/`tia_class`[/`tia_score`]. *done-when:* column populated for a real/sample run. *deps:* T1.3, T1.4a.
- **T1.5 (optional) Loop** — *deliverable:* filter-high-TIA + re-run set-cover driver + report (coverage retained vs TIA reduced). *done-when:* report produced. *deps:* T1.4b.

### Phase 2 — Algorithm 2 (mostly sequential; can start scaffolding after T0.2)
- **T2.1 Preprocess TIA** — *files:* `preprocess_kinetics.py`. *deliverable:* curated TSV gains a TIA column. *done-when:* schema + tests pass. *deps:* T0.2.
- **T2.2 Parser exposes TIA** — *files:* `data_parser.py`. *deps:* T2.1.
- **T2.3 Model head** — *files:* `model.py` (approach **a** second head; keep **b** relabel as fallback). *deps:* T2.2.
- **T2.4 Retrain** — *files:* `train_model.py` (multi-task loss / class weights / relabel path). *deliverable:* retrained classifier+regressor + metrics. *deps:* T2.3.
- **T2.5 Objective integration** — *files:* `guide_search.py`, `adapt_design.py` (combined `activity − λ·TIA`, tunable `λ`). *deps:* T2.4.
- **T2.6 On-target regression guard** — *deliverable:* confirm retrain does **not** regress vs README baselines (classifier acc ≈ 85.9% / AUC-ROC ≈ 0.872 / AUC-PR ≈ 0.972; regressor MSE ≈ 0.61 / Pearson ≈ 0.58). *done-when:* metrics ≥ baselines (or regression documented). *deps:* T2.4, T2.5.

### Phase 3 — Compare
- **T3.1 Head-to-head** — baseline ADAPT vs Algo 1 vs Algo 2 on a fixed input: % high-TIA guides removed, on-target activity retained, target coverage retained. *deliverable:* comparison report + table. *deps:* T1.4b, T2.5.

### Suggested fan-out
- Wave 1: T0.1 ∥ T0.2.
- Wave 2 (after T0.2): T1.1 ∥ T1.2 ∥ T1.4a ∥ T2.1.
- Wave 3: T1.3 → T1.4b; T2.2 → T2.3 → T2.4.
- Wave 4: T1.5, T2.5, T2.6 → T3.1.

---

## 6. Guardrails for all subagents

- **Never fabricate data or metrics.** Fixture results are labeled synthetic; missing data → BLOCKED-ON-DATA, not invention.
- **Additive first.** Algorithm 1 must not modify ADAPT's trained weights. Algorithm 2 changes are gated by the T2.6 regression guard.
- **Reuse existing encoders/splits** (`data_parser.py`, position-based non-overlapping split) to prevent leakage — do not invent a new encoding without reason.
- **Validate before committing:** run `pytest` and the repo's fast checks; re-read your diff adversarially.
- **Work on the branch your harness assigns; do not push to an unrelated branch. Do not open a PR unless the user asks.**
- **Confirm irreversible / outward-facing actions** with the user.

---

## 7. Global acceptance criteria

- Algo 1: a reusable TIA classifier with reported held-out metrics **and** ADAPT output carrying the new TIA column.
- Algo 2: a retrained TIA-aware model that demonstrably drops high-TIA guides at design time **without** regressing on-target metrics below README baselines.
- Phase 3: a comparison report quantifying, for a fixed input, high-TIA removed vs coverage/on-target retained across baseline / Algo 1 / Algo 2.

---

## 8. Open questions to raise with the user (from §2)
- Exact definitions of **TIA** and **TRACE** (metric, units, assay).
- Label **binary or continuous**?
- **Location** of the TIA/TRACE dataset and its **join key** to ADAPT guides.
- **Overlap** with the 19,209-pair on-target set.
- Preferred **operating point** (hard-omit vs tunable `λ`).
