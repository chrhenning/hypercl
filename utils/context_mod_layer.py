#!/usr/bin/env python3
# Copyright 2019 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
- **title**          :utils/context_mod_layer.py
- **author**         :ch
- **contact**        :henningc@ethz.ch
- **created**        :10/18/2019
- **version**        :1.0
- **python_version** :3.6.8

This module should represent a special gain-modulation layer that can modulate
neural computation based on an external context.
"""
import numpy as np
import torch
import torch.nn as nn

class ContextModLayer(nn.Module):
    r"""Implementation of a layer that can apply context-dependent modulation on
    the level of neuronal computation.

    The layer consists of two parameter vectors: gains :math:`\mathbf{g}`
    and shifts :math:`\mathbf{s}`, whereas gains represent a multiplicative
    modulation of input activations and shifts an additive modulation,
    respectively.

    Note, the weight vectors :math:`\mathbf{g}` and :math:`\mathbf{s}` might
    also be passed to the :meth:`forward` method, where one may pass a separate
    set of parameters for each sample in the input batch.

    Example:
        Assume that a :class:`ContextModLayer` is applied between a linear
        (fully-connected) layer
        :math:`\mathbf{y} \equiv W \mathbf{x} + \mathbf{b}` with input
        :math:`\mathbf{x}` and a nonlinear activation function
        :math:`z \equiv \sigma(y)`.

        The layer-computation in such a case will become

        .. math::

            \sigma \big( (W \mathbf{x} + \mathbf{b}) \odot \mathbf{g} + \
            \mathbf{s} \big)

    Attributes:
        weights: A list of all internal weights of this layer. If all weights
            are assumed to be generated externally, then this attribute will be
            ``None``.
        param_shapes: A list of list of integers. Each list represents the
            shape of a parameter tensor. Note, this attribute is
            independent of the attribute :attr:`weights`, it always comprises
            the shapes of all weight tensors as if the network would be stand-
            alone (i.e., no weights being passed to the :meth:`forward` method).

            .. note::
                The weights passed to the :meth:`forward` method might deviate
                from these shapes, as we allow passing a distinct set of
                parameters per sample in the input batch.
        num_ckpts (int): The number of existing weight checkpoints (i.e., how
            often the method :meth:`checkpoint_weights` was called).

    Args:
        num_features (int or tuple): Number of units in the layer (size of
            parameter vectors :math:`\mathbf{g}` and :math:`\mathbf{s}`).

            In case a ``tuple`` of integers is provided, the gain
            :math:`\mathbf{g}` and shift :math:`\mathbf{s}` parameters will
            become multidimensional tensors with the shape being prescribed
            by ``num_features``. Please note the `broadcasting rules`_ as
            :math:`\mathbf{g}` and :math:`\mathbf{s}` are simply multiplied
            or added to the input.

            Example:
                Consider the output of a convolutional layer with output shape
                ``[B,C,W,H]``. In case there should be a scalar gain and shift
                per feature map, ``num_features`` could be ``[C,1,1]`` or
                ``[1,C,1,1]`` (one might also pass a shape ``[B,C,1,1]`` to the
                :meth:`forward` method to apply separate shifts and gains per
                sample in the batch).

                Alternatively, one might want to provide shift and gain per
                output unit, i.e., ``num_features`` should be ``[C,W,H]``. Note,
                that due to weight sharing, all output activities within a
                feature map are computed using the same weights, which is why it
                is common practice to share shifts and gains within a feature
                map (e.g., in Spatial Batch-Normalization).
        no_weights (bool): If ``True``, the layer will have no trainable weights
            (:math:`\mathbf{g}` and :math:`\mathbf{s}`). Hence, weights are
            expected to be passed to the :meth:`forward` method.
        no_gains (bool): If ``True``, no gain parameters :math:`\mathbf{g}` will
            be modulating the input activity.

            .. note::
                Arguments ``no_gains`` and ``no_shifts`` might not be activated
                simultaneously!
        no_shifts (bool): If ``True``, no shift parameters :math:`\mathbf{s}`
            will be modulating the input activity.
        apply_gain_offset (bool, optional): If activated, this option will apply
            a constant offset of 1 to all gains, i.e., the computation becomes

            .. math::

                \sigma \big( (W \mathbf{x} + \mathbf{b}) \odot \
                (1 + \mathbf{g}) + \mathbf{s} \big)

            When could that be useful? In case the gains and shifts are
            generated by the same hypernetwork, a meaningful initialization
            might be difficult to achieve (e.g., such that gains are close to 1
            and shifts are close to 0 at the beginning). Therefore, one might
            initialize the hypernetwork such that all outputs are close to zero
            at the beginning and the constant shift ensures that meaningful
            gains are applied.
    .. _broadcasting rules:
        https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-\
        semantics
    """
    def __init__(self, num_features, no_weights=False, no_gains=False,
                 no_shifts=False, apply_gain_offset=False):
        super(ContextModLayer, self).__init__()

        raise NotImplementedError('Implementation not publicly available yet!')

if __name__ == '__main__':
    pass


