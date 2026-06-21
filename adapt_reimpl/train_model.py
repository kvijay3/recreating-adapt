"""Training loop for the Cas13a CNN model.

Trains two separate models following the paper exactly:
1. Classifier (BCE loss) — active/inactive, using paper's best hyperparameters
2. Regressor (MSE loss) — activity score on active pairs only

Paper: Metsky HC et al., Nature Biotechnology (2022)
doi:10.1038/s41587-022-01213-5
"""

import argparse
import os
import pickle

import numpy as np
import scipy.stats
import sklearn.metrics
import tensorflow as tf

from adapt_reimpl.data_parser import Cas13ActivityParser
from adapt_reimpl.model import (
    construct_model, classifier_params, regressor_params)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Cas13a CNN model for guide activity prediction")
    parser.add_argument('--data', required=True,
                        help="Path to TSV (or .tsv.gz) training data")
    parser.add_argument('--output', required=True,
                        help="Output directory for trained models")
    parser.add_argument('--context-nt', type=int, default=10)
    parser.add_argument('--max-num-epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=10,
                        help="Early stopping patience (paper uses 10)")
    parser.add_argument('--test-split-frac', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use-paper-params', action='store_true',
                        default=True,
                        help="Use paper's best hyperparameters (default)")
    return parser.parse_args()


def set_seed(seed):
    """Set TensorFlow and numpy seeds."""
    tf.random.set_seed(seed)
    np.random.seed(seed)


def read_data(args, classify=False, regress_only_on_active=False):
    """Read and prepare training data.

    Follows the paper's data split: 70% train+val (2/3 train, 1/3 val),
    30% test. Stratified by guide position. Uses exp-and-pos subset.

    Args:
        args: argument namespace
        classify: if True, prepare classification data
        regress_only_on_active: if True, only use active pairs for regression

    Returns:
        (x_train, y_train, x_val, y_val, x_test, y_test, data_parser)
    """
    test_frac = args.test_split_frac
    train_frac = (1.0 - test_frac) * (2.0 / 3.0)
    validation_frac = (1.0 - test_frac) * (1.0 / 3.0)

    data_parser = Cas13ActivityParser(
        tsv_path=args.data,
        subset='exp-and-pos',
        context_nt=args.context_nt,
        split=(train_frac, validation_frac, test_frac),
        shuffle_seed=args.seed,
        stratify_by_pos=True)
    data_parser.set_activity_mode(
        classify, not classify and not regress_only_on_active,
        regress_only_on_active)
    data_parser.read()

    x_train, y_train = data_parser.train_set()
    x_val, y_val = data_parser.validate_set()
    x_test, y_test = data_parser.test_set()

    print(f"DATA SIZES - Train: {len(x_train)}, "
          f"Validate: {len(x_val)}, Test: {len(x_test)}")

    return (x_train, y_train, x_val, y_val, x_test, y_test, data_parser)


def compute_test_metrics_classification(model, x_test, y_test,
                                         batch_size=216):
    """Compute test metrics matching the paper.

    Reports: loss (BCE), accuracy, AUC-ROC, AUC-PR.

    Args:
        model: trained classifier model
        x_test: test inputs
        y_test: test labels
        batch_size: batch size for evaluation

    Returns:
        dict of test metrics
    """
    y_pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_true = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    bce = sklearn.metrics.log_loss(y_true, y_pred_flat)
    accuracy = sklearn.metrics.accuracy_score(y_true,
                                               (y_pred_flat > 0.5).astype(int))
    auc_roc = sklearn.metrics.roc_auc_score(y_true, y_pred_flat)
    auc_pr = sklearn.metrics.average_precision_score(y_true, y_pred_flat)

    metrics = {
        'loss': bce,
        'bce': bce,
        'accuracy': accuracy,
        'auc-roc': auc_roc,
        'auc-pr': auc_pr,
    }
    return metrics


def compute_test_metrics_regression(model, x_test, y_test,
                                     batch_size=229):
    """Compute test metrics matching the paper.

    Reports: loss (MSE), MSE, MAE, r-Pearson, r-Spearman.

    Args:
        model: trained regressor model
        x_test: test inputs
        y_test: test labels
        batch_size: batch size for evaluation

    Returns:
        dict of test metrics
    """
    y_pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_true = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    mse = sklearn.metrics.mean_squared_error(y_true, y_pred_flat)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred_flat)

    r_pearson, _ = scipy.stats.pearsonr(y_true, y_pred_flat)
    rho_spearman, _ = scipy.stats.spearmanr(y_true, y_pred_flat)

    metrics = {
        'loss': mse,
        'mse': mse,
        'mae': mae,
        'r-pearson': r_pearson,
        'r-spearman': rho_spearman,
    }
    return metrics


def train_classifier(args):
    """Train the classification model using paper hyperparameters.

    Uses classifier_params() from model.py which contains the exact
    hyperparameters from the paper's best model (model-51373185),
    selected via random hyperparameter search with nested CV.

    Args:
        args: argument namespace

    Returns:
        (trained model, params, data_parser)
    """
    set_seed(args.seed)

    x_train, y_train, x_val, y_val, x_test, y_test, dp = read_data(
        args, classify=True)

    # Use paper's exact hyperparameters
    params = classifier_params()
    params['context_nt'] = args.context_nt
    params['max_num_epochs'] = args.max_num_epochs

    model = construct_model(params, x_train.shape, regression=False)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=params['learning_rate'])
    loss = 'binary_crossentropy'
    metrics = ['accuracy']

    # Class weights (balanced, as in paper)
    import sklearn.utils.class_weight as cw
    y_train_labels = [int(y_train[i][0]) for i in range(len(y_train))]
    classes = sorted(np.unique(y_train_labels))
    class_weight = cw.compute_class_weight(
        class_weight='balanced',
        classes=np.array(classes),
        y=y_train_labels)
    class_weight_dict = {i: w for i, w in enumerate(class_weight)}

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Early stopping with patience=10 (paper uses 10)
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=args.patience,
        restore_best_weights=True)

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=params['batch_size'],
        callbacks=[es],
        class_weight=class_weight_dict,
        epochs=args.max_num_epochs,
        verbose=2)

    # Compute paper-style test metrics
    test_metrics = compute_test_metrics_classification(
        model, x_test, y_test, batch_size=params['batch_size'])
    print(f"TEST METRICS: {test_metrics}")

    return model, params, dp


def train_regressor(args):
    """Train the regression model on active pairs only.

    Uses regressor_params() from model.py which contains the exact
    hyperparameters from the paper's best regressor (model-f8b6fd5d).
    Trained with regress_only_on_active=True, meaning only guide-target
    pairs with activity > -4.0 are used for training.

    Args:
        args: argument namespace

    Returns:
        (trained model, params, data_parser)
    """
    set_seed(args.seed)

    x_train, y_train, x_val, y_val, x_test, y_test, dp = read_data(
        args, regress_only_on_active=True)

    # Use paper's exact hyperparameters
    params = regressor_params()
    params['context_nt'] = args.context_nt
    params['max_num_epochs'] = args.max_num_epochs

    model = construct_model(params, x_train.shape, regression=True)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=params['learning_rate'])
    loss = 'mse'
    metrics = ['mse', 'mae']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Early stopping with patience=10 (paper uses 10)
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=args.patience,
        restore_best_weights=True)

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=params['batch_size'],
        callbacks=[es],
        epochs=args.max_num_epochs,
        verbose=2)

    # Compute paper-style test metrics
    if len(x_test) > 0:
        test_metrics = compute_test_metrics_regression(
            model, x_test, y_test, batch_size=params['batch_size'])
        print(f"TEST METRICS: {test_metrics}")

    return model, params, dp


def save_model(model, params, output_dir, model_type, guide_length=28,
               default_threshold=None):
    """Save model as TF SavedModel with assets.extra metadata.

    Args:
        model: trained model
        params: parameter dict
        output_dir: base output directory
        model_type: 'classification' or 'regression'
        guide_length: guide length
        default_threshold: default threshold for the model
    """
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)

    # Save model weights (Keras 3 compatible for custom models)
    model_path = os.path.join(model_dir, 'model.weights.h5')
    model.save_weights(model_path)

    # Save params so model can be reconstructed
    with open(os.path.join(model_dir, 'model.params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # Save assets.extra
    assets_dir = os.path.join(model_dir, 'assets.extra')
    os.makedirs(assets_dir, exist_ok=True)

    with open(os.path.join(assets_dir, 'context_nt.arg'), 'w') as f:
        f.write(str(params['context_nt']) + '\n')
    with open(os.path.join(assets_dir, 'guide_length.arg'), 'w') as f:
        f.write(str(guide_length) + '\n')
    if default_threshold is not None:
        with open(os.path.join(assets_dir, 'default_threshold.arg'), 'w') as f:
            f.write(str(default_threshold) + '\n')

    print(f"Saved {model_type} model to {model_dir}")


def main():
    """Main training entry point.

    Trains both classifier and regressor using paper-exact hyperparameters.
    Classifier: BCE loss, balanced class weights, patience=10 early stopping.
    Regressor: MSE loss on active pairs only, patience=10 early stopping.
    Both use the paper's best models from random hyperparameter search.
    """
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Train classifier
    print("=" * 60)
    print("Training CLASSIFIER model")
    print("=" * 60)
    cls_model, cls_params, cls_dp = train_classifier(args)
    save_model(cls_model, cls_params, args.output, 'classification',
               default_threshold=0.577)

    # Train regressor
    print("=" * 60)
    print("Training REGRESSOR model")
    print("=" * 60)
    reg_model, reg_params, reg_dp = train_regressor(args)
    save_model(reg_model, reg_params, args.output, 'regression')

    print("\nTraining complete. Models saved to:", args.output)


if __name__ == '__main__':
    main()
