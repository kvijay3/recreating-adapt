"""CNN model for predicting Cas13a guide activity.

Reimplements CasCNNWithParallelFilters as a tf.keras.Model subclass.
"""

import numpy as np
import tensorflow as tf


class LocallyConnected1D(tf.keras.layers.Layer):
    """Locally connected 1D layer (unshared weights convolution).

    Each position gets its own set of weights, unlike Conv1D which shares
    weights across positions. This is a reimplementation of the removed
    tf.keras.layers.LocallyConnected1D for Keras 3 compatibility.

    Args:
        filters: int, dimensionality of output space
        kernel_size: int, length of the 1D convolution window
        strides: int, stride of the convolution
        activation: str, activation function to use
        name: str, layer name
    """

    def __init__(self, filters, kernel_size, strides=1,
                 activation=None, name=None, **kwargs):
        super(LocallyConnected1D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # input_shape: (batch, width, channels)
        in_width = input_shape[1]
        in_channels = input_shape[2]

        # Compute number of output positions
        self.num_positions = (in_width - self.kernel_size) // self.strides + 1
        if self.num_positions <= 0:
            self.num_positions = 1

        # Each position has its own weight matrix and bias
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.num_positions, self.kernel_size * in_channels,
                   self.filters),
            initializer='glorot_uniform',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.num_positions, self.filters),
            initializer='zeros',
            trainable=True)

    def call(self, inputs):
        # inputs: (batch, width, channels)
        batch_size = tf.shape(inputs)[0]
        patches = []
        for i in range(self.num_positions):
            start = i * self.strides
            end = start + self.kernel_size
            patch = inputs[:, start:end, :]  # (batch, kernel_size, channels)
            patch_flat = tf.reshape(
                patch, (batch_size, -1))  # (batch, kernel_size * channels)
            # Apply position-specific weights
            out = tf.matmul(patch_flat, self.kernel[i]) + self.bias[i]
            patches.append(out)

        output = tf.stack(patches, axis=1)  # (batch, num_positions, filters)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_positions, self.filters)

    def get_config(self):
        config = super(LocallyConnected1D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': tf.keras.activations.serialize(self.activation),
        })
        return config


class CasCNNWithParallelFilters(tf.keras.Model):
    """CNN with parallel convolutional filters of different widths.

    Architecture:
    - Input: (batch, 2*context_nt + guide_length, 8) — 8-channel one-hot
    - Parallel Conv1D groups with configurable filter widths
    - Each followed by BatchNorm and MaxPooling1D
    - Optional LocallyConnected1D layers
    - Groups concatenated along width axis
    - Flatten → optional GC content feature → FC layers → output
    """

    def __init__(self, params, regression):
        """Initialize the model.

        Args:
            params: dict of hyperparameters with keys:
                - conv_filter_width: list of int or None
                - conv_num_filters: int
                - pool_window_width: int
                - pool_strategy: 'max', 'avg', or 'max-and-avg'
                - fully_connected_dim: list of int
                - dropout_rate: float
                - l2_factor: float
                - activation_fn: str ('relu' or 'elu')
                - add_gc_content: bool
                - context_nt: int
                - batch_size: int
                - learning_rate: float
                - sample_weight_scaling_factor: float
                - skip_batch_norm: bool
                - locally_connected_width: list of int or None
                - locally_connected_dim: int
                - regression_clip: bool
                - regression_clip_alpha: float
            regression: if True, regression (linear output); else classification (sigmoid)
        """
        super(CasCNNWithParallelFilters, self).__init__()

        self.regression = regression
        self.add_gc_content = params.get('add_gc_content', False)
        self.context_nt = params.get('context_nt', 10)

        if params.get('sample_weight_scaling_factor', 0) < 0:
            raise ValueError("sample_weight_scaling_factor must be >= 0")
        self.sample_weight_scaling_factor = params.get(
            'sample_weight_scaling_factor', 0)

        self.batch_size = params.get('batch_size', 32)
        self.learning_rate = params.get('learning_rate', 1e-5)

        if self.add_gc_content:
            self.guide_slice = tf.keras.layers.Cropping1D(
                (self.context_nt, self.context_nt))

        if params.get('conv_filter_width') is None:
            conv_filter_widths = [None]
        else:
            conv_filter_widths = params['conv_filter_width']

        self.convs = []
        self.batchnorms = []
        self.pools = []
        self.pools_2 = []
        self.lcs = []

        for filter_width in conv_filter_widths:
            if filter_width is not None:
                conv_layer_num_filters = params.get('conv_num_filters', 20)
                conv = tf.keras.layers.Conv1D(
                    conv_layer_num_filters,
                    filter_width,
                    strides=1,
                    padding='valid',
                    activation=params.get('activation_fn', 'relu'),
                    name='group_w' + str(filter_width) + '_conv')

                if params.get('skip_batch_norm', False):
                    batchnorm = None
                else:
                    batchnorm = tf.keras.layers.BatchNormalization(
                        name='group_w' + str(filter_width) + '_batchnorm')

                pool_window_width = params.get('pool_window_width', 2)
                pool_stride = max(1, int(pool_window_width / 2))
                maxpool = tf.keras.layers.MaxPooling1D(
                    pool_size=pool_window_width,
                    strides=pool_stride,
                    name='group_w' + str(filter_width) + '_maxpool')
                avgpool = tf.keras.layers.AveragePooling1D(
                    pool_size=pool_window_width,
                    strides=pool_stride,
                    name='group_w' + str(filter_width) + '_avgpool')

                self.convs.append(conv)
                self.batchnorms.append(batchnorm)

                pool_strategy = params.get('pool_strategy', 'max')
                if pool_strategy == 'max':
                    self.pools.append(maxpool)
                    self.pools_2.append(None)
                elif pool_strategy == 'avg':
                    self.pools.append(avgpool)
                    self.pools_2.append(None)
                elif pool_strategy == 'max-and-avg':
                    self.pools.append(maxpool)
                    self.pools_2.append(avgpool)
                else:
                    raise Exception("Unknown pool_strategy")
            else:
                self.convs.append(None)
                self.batchnorms.append(None)
                self.pools.append(None)
                self.pools_2.append(None)

            # Locally connected layers
            if params.get('locally_connected_width') is not None:
                lcs_for_conv = []
                locally_connected_dim = params.get('locally_connected_dim', 1)
                for i, lc_width in enumerate(params['locally_connected_width']):
                    if filter_width is not None:
                        name = 'group_w' + str(filter_width) + '_lc_w' + str(lc_width)
                    else:
                        name = 'lc_w' + str(lc_width)
                    stride = max(1, int(lc_width / 2))
                    lc = LocallyConnected1D(
                        locally_connected_dim,
                        lc_width,
                        strides=stride,
                        activation=params.get('activation_fn', 'relu'),
                        name=name)
                    lcs_for_conv.append(lc)
                self.lcs.append(lcs_for_conv)
            else:
                self.lcs.append(None)

        if conv_filter_widths != [None] and params.get('pool_strategy') == 'max-and-avg':
            self.pool_merge = tf.keras.layers.Concatenate(axis=1, name='merge_pool')

        if params.get('locally_connected_width') is not None:
            if len(params['locally_connected_width']) > 1:
                self.lc_merge = tf.keras.layers.Concatenate(
                    axis=1, name='merge_lc')

        if len(conv_filter_widths) > 1:
            self.merge = tf.keras.layers.Concatenate(
                axis=1, name='merge_groups')

        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers
        self.dropouts = []
        self.fcs = []
        for i, fc_hidden_dim in enumerate(params.get('fully_connected_dim', [20])):
            dropout = tf.keras.layers.Dropout(
                params.get('dropout_rate', 0.25),
                name='dropout_' + str(i + 1))
            fc = tf.keras.layers.Dense(
                fc_hidden_dim,
                activation=params.get('activation_fn', 'relu'),
                name='fc_' + str(i + 1))
            self.dropouts.append(dropout)
            self.fcs.append(fc)

        # Final layer
        fc_final_dim = 1
        if regression:
            final_activation = 'linear'
        else:
            final_activation = 'sigmoid'
        self.fc_final = tf.keras.layers.Dense(
            fc_final_dim, activation=final_activation, name='fc_final')

        # Regression clip
        if (regression and params.get('regression_clip', False)):
            min_out = -4
            def clip(x):
                return tf.keras.activations.relu(x - min_out) + min_out
            self.clip_output = clip
            alpha = params.get('regression_clip_alpha', 0.1)
            def clip_leaky(x):
                return tf.nn.leaky_relu(
                    x - min_out, alpha=alpha) + min_out
            self.clip_output_leaky = clip_leaky
        else:
            self.clip_output = None

        # L2 regularization
        l2_factor = params.get('l2_factor', 0)
        if l2_factor > 0:
            l2_regularizer = tf.keras.regularizers.l2(l2_factor)
            for layer in self.layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = l2_regularizer

    def call(self, x, training=False):
        """Forward pass.

        Args:
            x: input tensor of shape (batch, seq_len, 8)
            training: whether in training mode

        Returns:
            output tensor of shape (batch, 1)
        """
        if self.add_gc_content:
            x_guide_region = self.guide_slice(x)
            x_guide = tf.slice(x_guide_region,
                               [0, 0, 4],
                               [-1, -1, 4])
            base_count = tf.reduce_sum(x_guide, axis=1, keepdims=True)
            gc_count = base_count[:, :, 1] + base_count[:, :, 2]
            guide_len = tf.cast(tf.shape(x_guide)[1], tf.float32)
            gc_content = gc_count / guide_len

        group_outputs = []
        for conv, batchnorm, pool_1, pool_2, lcs in zip(
                self.convs, self.batchnorms, self.pools, self.pools_2, self.lcs):
            if conv is not None:
                group_x = conv(x)
                if batchnorm is not None:
                    group_x = batchnorm(group_x, training=training)

                if pool_2 is None:
                    group_x = pool_1(group_x)
                else:
                    group_x_1 = pool_1(group_x)
                    group_x_2 = pool_2(group_x)
                    group_x = self.pool_merge([group_x_1, group_x_2])
            else:
                group_x = x

            if lcs is not None:
                if len(lcs) == 1:
                    group_x = lcs[0](group_x)
                else:
                    lc_outputs = []
                    for lc in lcs:
                        lc_outputs.append(lc(group_x))
                    group_x = self.lc_merge(lc_outputs)

            group_outputs.append(group_x)

        if len(group_outputs) == 1:
            x = group_outputs[0]
        else:
            x = self.merge(group_outputs)
        x = self.flatten(x)

        if self.add_gc_content:
            x = tf.concat([x, gc_content], -1)

        for dropout, fc in zip(self.dropouts, self.fcs):
            x = dropout(x, training=training)
            x = fc(x)

        x = dropout(x, training=training)
        x = self.fc_final(x)

        if self.clip_output is not None:
            if training:
                x = self.clip_output_leaky(x)
            else:
                x = self.clip_output(x)
        return x


def construct_model(params, shape, regression=False):
    """Construct and build a CasCNNWithParallelFilters model.

    Args:
        params: dict of hyperparameters
        shape: shape of input data (for building)
        regression: if True, regression; else classification

    Returns:
        CasCNNWithParallelFilters model
    """
    model = CasCNNWithParallelFilters(params, regression)
    model.build(shape)
    return model


def default_params():
    """Return default hyperparameters.

    These are generic defaults; for paper-exact parameters, use
    classifier_params() or regressor_params() instead.

    Returns:
        dict of default parameters
    """
    return {
        'conv_filter_width': [3, 5, 7],
        'conv_num_filters': 20,
        'pool_window_width': 2,
        'pool_strategy': 'max',
        'fully_connected_dim': [20],
        'dropout_rate': 0.25,
        'l2_factor': 0,
        'activation_fn': 'relu',
        'add_gc_content': False,
        'context_nt': 10,
        'batch_size': 32,
        'learning_rate': 1e-5,
        'sample_weight_scaling_factor': 0,
        'skip_batch_norm': False,
        'locally_connected_width': None,
        'locally_connected_dim': 1,
        'regression_clip': True,
        'regression_clip_alpha': 0.1,
    }


def classifier_params():
    """Return hyperparameters matching the paper's best classifier model.

    From adapt-seq-design models/cas13/classify/model-51373185.
    Selected via random hyperparameter search (50 samples) with
    nested cross-validation.

    Returns:
        dict of classifier parameters
    """
    return {
        'activation_fn': 'relu',
        'add_gc_content': False,
        'batch_size': 216,
        'context_nt': 10,
        'conv_filter_width': [1],
        'conv_num_filters': 238,
        'dropout_rate': 0.3300867687463425,
        'fully_connected_dim': [46],
        'l2_factor': 2.2763059253307743e-06,
        'learning_rate': 8.266299287317161e-05,
        'locally_connected_dim': 4,
        'locally_connected_width': [1, 2],
        'max_num_epochs': 1000,
        'pool_strategy': 'avg',
        'pool_window_width': 2,
        'regression': False,
        'sample_weight_scaling_factor': 0,
        'skip_batch_norm': False,
    }


def regressor_params():
    """Return hyperparameters matching the paper's best regressor model.

    From adapt-seq-design models/cas13/regress/model-f8b6fd5d.
    Selected via random hyperparameter search with nested cross-validation.
    Trained with regress_only_on_active=True (only active guide-target pairs).

    Returns:
        dict of regressor parameters
    """
    return {
        'activation_fn': 'relu',
        'add_gc_content': False,
        'batch_size': 229,
        'context_nt': 10,
        'conv_filter_width': [1, 2],
        'conv_num_filters': 25,
        'dropout_rate': 0.29845230312675486,
        'fully_connected_dim': [53],
        'l2_factor': 2.6309419773217563e-06,
        'learning_rate': 0.0017914755444431493,
        'locally_connected_dim': 3,
        'locally_connected_width': [1, 2],
        'max_num_epochs': 1000,
        'pool_strategy': 'avg',
        'pool_window_width': 2,
        'regression': True,
        'sample_weight_scaling_factor': 0,
        'skip_batch_norm': True,
    }
