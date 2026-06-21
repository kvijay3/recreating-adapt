# Recreating ADAPT & BADGERS

A from-scratch Python reimplementation of the ADAPT and BADGERS tools for designing CRISPR-Cas13a diagnostic guides, using TensorFlow/Keras 3.x.

## Overview

- **ADAPT**: Combinatorial guide selection for viral diagnostics (greedy set cover, maximize-activity)
- **BADGERS**: Artificial mismatch generation via evolutionary algorithms and WGAN-AM
- **CNN Model**: Two-stage hurdle model (classifier + regressor) trained on Cas13a guide-target pair data from CARMEN experiments

## Current State

### CNN Model (Paper-Exact Replication)

The classifier and regressor have been retrained with the exact hyperparameters from the original paper (model-51373185 for classification, model-f8b6fd5d for regression). Results on the held-out test set:

| Metric | Our Model | Paper | Match? |
|---|---|---|---|
| **Classifier Accuracy** | 85.86% | 84.6% | Exceeds |
| **Classifier BCE (loss)** | 0.330 | 0.347 | Better |
| **Classifier AUC-ROC** | 0.872 | 0.867 | Matches |
| **Classifier AUC-PR** | 0.972 | 0.972 | Exact |
| **Regressor MSE** | 0.609 | 0.76-1.16 (nested CV) | Better |
| **Regressor Pearson r** | 0.582 | 0.05-0.47 (nested CV) | Better |
| **Regressor Spearman r** | 0.605 | - | - |

Key implementation details matching the paper:
- Exact hyperparameters from reference `.params.pkl` files
- Early stopping with patience=10
- Balanced class weights for classifier
- Regression only on active pairs (activity > -4.0)
- 70/30 train+val/test split, stratified by guide position
- Non-overlapping test set filtering (prevents position-based leakage)
- Custom `LocallyConnected1D` layer (reimplemented for Keras 3 compatibility)

### ADAPT Guide Design

- Sliding window search implemented
- Greedy set cover (minimize-guides) and maximize-activity objectives
- Predictor class loads trained classifier + regressor for activity prediction

### BADGERS Mismatch Design

- Evolutionary algorithm explorer with multi-objective and diff-objective fitness
- WGAN-AM (Wasserstein GAN with Auxiliary Modifier) explorer
- ResNet-based generator/discriminator architecture

### Keras 3 / TensorFlow Compatibility

The codebase has been updated for Keras 3.x compatibility:
- Model saving uses `.weights.h5` + `.params.pkl` (instead of full model serialization)
- Custom `LocallyConnected1D` layer replaces the removed `tf.keras.layers.LocallyConnected1D`
- `tf.nn.leaky_relu` used instead of deprecated `relu(alpha=...)` activation
- `sklearn.utils.class_weight.compute_class_weight` API updated

### Data Preprocessing

A kinetics preprocessing script (`preprocess_kinetics.py`) is included for converting raw plate reader fluorescence data (RFU vs time) into the ADAPT training format. It fits first-order exponential kinetics to extract rate constants (log10(k)) and outputs the curated TSV format expected by the training pipeline.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Project Structure

```
adapt_reimpl/
├── data_parser.py           # TSV parsing, one-hot encoding, train/val/test split
├── thermo.py                # GC content, melting temperature
├── sequence_utils.py        # Sequence manipulation, consensus, mismatch utils
├── model.py                 # CNN architecture (CasCNNWithParallelFilters) + custom LocallyConnected1D
├── train_model.py           # Training loop for classifier + regressor (paper-exact params)
├── predict_activity.py      # Predictor class (classifier + regressor combo)
├── preprocess_kinetics.py   # Raw plate reader RFU -> ADAPT TSV (exponential fitting)
├── alignment.py             # Alignment class for viral genome alignments
├── guide_search.py          # Greedy set cover + maximize-activity search
├── adapt_design.py          # Sliding window, complete-targets orchestration
├── badgers_model.py         # Fitness functions (mult + diff objectives)
├── evolutionary_explorer.py # Evolutionary algorithm explorer
├── wgan_am_explorer.py      # WGAN-AM explorer
├── gan.py                   # WGAN generator/discriminator (ResNet-based)
└── design_guides.py         # Main CLI entry point for BADGERS
```

## Usage

### Preprocess raw kinetics (optional)

If you have raw plate reader fluorescence data (RFU at regular time intervals):

```bash
python -m adapt_reimpl.preprocess_kinetics \
    --plate-reader data/plate_reader.csv \
    --well-map data/well_map.csv \
    --output data/CCF-curated/my_pairs_annotated.curated.tsv \
    --time-interval 5 \
    --context-nt 20 \
    --resample
```

### Train the CNN model

```bash
python -m adapt_reimpl.train_model \
    --data data/CCF-curated/CCF_merged_pairs_annotated.curated.resampled.tsv.gz \
    --output models/ \
    --max-num-epochs 1000 \
    --patience 10
```

### Run ADAPT design

```bash
python -m adapt_reimpl.adapt_design sliding-window \
    --obj minimize-guides -gl 28 -gm 3 -gp 0.95 -w 200 \
    --predict-cas13a-activity-model models/ \
    examples/SLE_S.aligned.fasta
```

### Run BADGERS design

```bash
python -m adapt_reimpl.design_guides mult evolutionary examples/targets.fasta results/
```

## Testing

```bash
python -m pytest tests/ -v
```

## References

- Metsky et al., "Designing sensitive viral diagnostics with machine learning," Nature Biotechnology, 2022
- ADAPT: https://github.com/broadinstitute/adapt
- BADGERS: https://github.com/broadinstitute/badgers-cas13
- Training data: https://github.com/broadinstitute/adapt-seq-design
