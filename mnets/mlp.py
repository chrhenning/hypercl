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
- **title**          :mnets/mlp.py
- **author**         :ch
- **contact**        :henningc@ethz.ch
- **created**        :10/21/2019
- **version**        :1.0
- **python_version** :3.6.8

Implementation of a fully-connected neural network.

An example usage is as a main model, that doesn't include any trainable weights.
Instead, weights are received as additional inputs. For instance, using an
auxilliary network, a so called hypernetwork, see

    Ha et al., "HyperNetworks", arXiv, 2016,
    https://arxiv.org/abs/1609.09106
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mnets.mnet_interface import MainNetInterface
from utils.batchnorm_layer import BatchNormLayer
from utils.context_mod_layer import ContextModLayer
from utils.torch_utils import init_params

class MLP(nn.Module, MainNetInterface):
    """Implementation of a Multi-Layer Perceptron (MLP).

    This is a simple fully-connected network, that receives input vector
    :math:`\mathbf{x}` and outputs a vector :math:`\mathbf{y}` of real values.

    The output mapping does not include a non-linearity by default, as we wanna
    map to the whole real line (but see argument ``out_fn``).

    Args:
        n_in (int): Number of inputs.
        n_out (int): Number of outputs.
        hidden_layers (list): A list of integers, each number denoting the size
            of a hidden layer.
        activation_fn: The nonlinearity used in hidden layers. If ``None``, no
            nonlinearity will be applied.
        use_bias (bool): Whether layers may have bias terms.
        no_weights (bool): If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.
        init_weights (optional): This option is for convinience reasons.
            The option expects a list of parameter values that are used to
            initialize the network weights. As such, it provides a
            convinient way of initializing a network with a weight draw
            produced by the hypernetwork.

            Note, internal weights (see 
            :attr:`mnets.mnet_interface.MainNetInterface.weights`) will be
            affected by this argument only.
        dropout_rate: If ``-1``, no dropout will be applied. Otherwise a number
            between 0 and 1 is expected, denoting the dropout rate of hidden
            layers.
        use_spectral_norm: Use spectral normalization for training.
        use_batch_norm (bool): Whether batch normalization should be used. Will
            be applied before the activation function in all hidden layers.
        bn_track_stats (bool): If batch normalization is used, then this option
            determines whether running statistics are tracked in these
            layers or not (see argument ``track_running_stats`` of class
            :class:`utils.batchnorm_layer.BatchNormLayer`).

            If ``False``, then batch statistics are utilized even during
            evaluation. If ``True``, then running stats are tracked. When
            using this network in a continual learning scenario with
            different tasks then the running statistics are expected to be
            maintained externally. The argument ``stats_id`` of the method
            :meth:`utils.batchnorm_layer.BatchNormLayer.forward` can be
            provided using the argument ``condition`` of method :meth:`forward`.

            Example:
                To maintain the running stats, one can simply iterate over
                all batch norm layers and checkpoint the current running
                stats (e.g., after learning a task when applying a Continual
                learning scenario).

                .. code:: python

                    for bn_layer in net.batchnorm_layers:
                        bn_layer.checkpoint_stats()
        distill_bn_stats (bool): If ``True``, then the shapes of the batchnorm
            statistics will be added to the attribute
            :attr:`mnets.mnet_interface.MainNetInterface.\
hyper_shapes_distilled` and the current statistics will be returned by the
            method :meth:`distillation_targets`.

            Note, this attribute may only be ``True`` if ``bn_track_stats``
            is ``True``.
        use_context_mod (bool): Add context-dependent modulation layers
            :class:`utils.context_mod_layer.ContextModLayer` after the linear
            computation of each layer.
        context_mod_inputs (bool): Whether context-dependent modulation should
            also be applied to network intpus directly. I.e., assume
            :math:`\mathbf{x}` is the input to the network. Then the first
            network operation would be to modify the input via
            :math:`\mathbf{x} \cdot \mathbf{g} + \mathbf{s}` using context-
            dependent gain and shift parameters.

            Note:
                Argument applies only if ``use_context_mod`` is ``True``.
        no_last_layer_context_mod (bool): If ``True``, context-dependent
            modulation will not be applied to the output layer.

            Note:
                Argument applies only if ``use_context_mod`` is ``True``.
        context_mod_no_weights (bool): The weights of the context-mod layers
            (:class:`utils.context_mod_layer.ContextModLayer`) are treated
            independently of the option ``no_weights``.
            This argument can be used to decide whether the context-mod
            parameters (gains and shifts) are maintained internally or
            externally.

            Note:
                Check out argument ``weights`` of the :meth:`forward` method
                on how to correctly pass weights to the network that are
                externally maintained.
        context_mod_post_activation (bool): Apply context-mod layers after the
            activation function (``activation_fn``) in hidden layer rather than
            before, which is the default behavior.

            Note:
                This option only applies if ``use_context_mod`` is ``True``.

            Note:
                This option does not affect argument ``context_mod_inputs``.

            Note:
                This option does not affect argument
                ``no_last_layer_context_mod``. Hence, if a output-nonlinearity
                is applied through argument ``out_fn``, then context-modulation
                would be applied before this non-linearity.
        context_mod_gain_offset (bool): Activates option ``apply_gain_offset``
            of class :class:`utils.context_mod_layer.ContextModLayer` for all
            context-mod layers that will be instantiated.
        out_fn (optional): If provided, this function will be applied to the
            output neurons of the network.

            Warning:
                This changes the interpretation of the output of the
                :meth:`forward` method.
        verbose (bool): Whether to print information (e.g., the number of
            weights) during the construction of the network.
    """
    def __init__(self, n_in=1, n_out=1, hidden_layers=[10, 10],
                 activation_fn=torch.nn.ReLU(), use_bias=True, no_weights=False,
                 init_weights=None, dropout_rate=-1, use_spectral_norm=False,
                 use_batch_norm=False, bn_track_stats=True,
                 distill_bn_stats=False, use_context_mod=False,
                 context_mod_inputs=False, no_last_layer_context_mod=False,
                 context_mod_no_weights=False,
                 context_mod_post_activation=False,
                 context_mod_gain_offset=False, out_fn=None, verbose=True):
        # FIXME find a way using super to handle multiple inheritence.
        #super(MainNetwork, self).__init__()
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        if use_spectral_norm:
            raise NotImplementedError('Spectral normalization not yet ' +
                                      'implemented for this network.')

        if use_batch_norm and use_context_mod:
            # FIXME Does it make sense to have both enabled?
            # I.e., should we produce a warning or error?
            pass

        self._a_fun = activation_fn
        assert(init_weights is None or \
               (not no_weights or not context_mod_no_weights))
        self._no_weights = no_weights
        self._dropout_rate = dropout_rate
        #self._use_spectral_norm = use_spectral_norm
        self._use_batch_norm = use_batch_norm
        self._bn_track_stats = bn_track_stats
        self._distill_bn_stats = distill_bn_stats and use_batch_norm
        self._use_context_mod = use_context_mod
        self._context_mod_inputs = context_mod_inputs
        self._no_last_layer_context_mod = no_last_layer_context_mod
        self._context_mod_no_weights = context_mod_no_weights
        self._context_mod_post_activation = context_mod_post_activation
        self._context_mod_gain_offset = context_mod_gain_offset
        self._out_fn = out_fn

        self._has_bias = use_bias
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True
        self._has_linear_out = True if out_fn is None else False

        if use_spectral_norm and no_weights:
            raise ValueError('Cannot use spectral norm in a network without ' +
                             'parameters.')

        # FIXME make sure that this implementation is correct in all situations
        # (e.g., what to do if weights are passed to the forward method?).
        if use_spectral_norm:
            self._spec_norm = nn.utils.spectral_norm
        else:
            self._spec_norm = lambda x : x # identity

        self._param_shapes = []
        self._weights = None if no_weights and context_mod_no_weights \
            else nn.ParameterList()
        self._hyper_shapes_learned = None \
            if not no_weights and not context_mod_no_weights else []

        if dropout_rate != -1:
            assert(dropout_rate >= 0. and dropout_rate <= 1.)
            self._dropout = nn.Dropout(p=dropout_rate)

        ### Define and initialize context mod weights.
        self._context_mod_layers = nn.ModuleList() if use_context_mod else None
        self._context_mod_shapes = [] if use_context_mod else None

        if use_context_mod:
            cm_ind = 0
            cm_sizes = []
            if context_mod_inputs:
                cm_sizes.append(n_in)
            cm_sizes.extend(hidden_layers)
            if not no_last_layer_context_mod:
                cm_sizes.append(n_out)

            for i, n in enumerate(cm_sizes):
                cmod_layer = ContextModLayer(n,
                    no_weights=context_mod_no_weights,
                    apply_gain_offset=context_mod_gain_offset)
                self._context_mod_layers.append(cmod_layer)

                self.param_shapes.extend(cmod_layer.param_shapes)
                self._context_mod_shapes.extend(cmod_layer.param_shapes)
                if context_mod_no_weights:
                    self._hyper_shapes_learned.extend(cmod_layer.param_shapes)
                else:
                    self._weights.extend(cmod_layer.weights)

                # FIXME ugly code. Move initialization somewhere else.
                if not context_mod_no_weights and init_weights is not None:
                    assert(len(cmod_layer.weights) == 2)
                    for ii in range(2):
                        assert(np.all(np.equal( \
                                list(init_weights[cm_ind].shape),
                                list(cm_ind.weights[ii].shape))))
                        cmod_layer.weights[ii].data = init_weights[cm_ind]
                        cm_ind += 1

            if init_weights is not None:
                init_weights = init_weights[cm_ind:]

        ### Define and initialize batch norm weights.
        self._batchnorm_layers = nn.ModuleList() if use_batch_norm else None

        if use_batch_norm:
            if distill_bn_stats:
                self._hyper_shapes_distilled = []

            bn_ind = 0
            for i, n in enumerate(hidden_layers):
                bn_layer = BatchNormLayer(n, affine=not no_weights,
                    track_running_stats=bn_track_stats)
                self._batchnorm_layers.append(bn_layer)
                self._param_shapes.extend(bn_layer.param_shapes)

                if no_weights:
                    self._hyper_shapes_learned.extend(bn_layer.param_shapes)
                else:
                    self._weights.extend(bn_layer.weights)

                if distill_bn_stats:
                    self._hyper_shapes_distilled.extend( \
                        [list(p.shape) for p in bn_layer.get_stats(0)])

                # FIXME ugly code. Move initialization somewhere else.
                if not no_weights and init_weights is not None:
                    assert(len(bn_layer.weights) == 2)
                    for ii in range(2):
                        assert(np.all(np.equal( \
                                list(init_weights[bn_ind].shape),
                                list(bn_layer.weights[ii].shape))))
                        bn_layer.weights[ii].data = init_weights[bn_ind]
                        bn_ind += 1

            if init_weights is not None:
                init_weights = init_weights[bn_ind:]

        # Compute shapes of linear layers.
        linear_shapes = MLP.weight_shapes(n_in=n_in, n_out=n_out,
            hidden_layers=hidden_layers, use_bias=use_bias)
        self._param_shapes.extend(linear_shapes)

        num_weights = MainNetInterface.shapes_to_num_weights(self._param_shapes)

        if verbose:
            if use_context_mod:
                cm_num_weights = 0
                for cm_layer in self._context_mod_layers:
                    cm_num_weights += MainNetInterface.shapes_to_num_weights( \
                        cm_layer.param_shapes)

            print('Creating an MLP with %d weights' % num_weights
                  + (' (including %d weights associated with-' % cm_num_weights
                     + 'context modulation)' if use_context_mod else '')
                  + '.'
                  + (' The network uses dropout.' if dropout_rate != -1 else '')
                  + (' The network uses batchnorm.' if use_batch_norm  else ''))

        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        if no_weights:
            self._hyper_shapes_learned.extend(linear_shapes)
            self._is_properly_setup()
            return

        ### Define and initialize linear weights.
        for i, dims in enumerate(linear_shapes):
            self._weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))
            if len(dims) == 1:
                self._layer_bias_vectors.append(self._weights[-1])
            else:
                self._layer_weight_tensors.append(self._weights[-1])

        if init_weights is not None:
            assert(len(init_weights) == len(linear_shapes))
            for i in range(len(init_weights)):
                assert(np.all(np.equal(list(init_weights[i].shape),
                                       linear_shapes[i])))
                if use_bias:
                    if i % 2 == 0:
                        self._layer_weight_tensors[i//2].data = init_weights[i]
                    else:
                        self._layer_bias_vectors[i//2].data = init_weights[i]
                else:
                    self._layer_weight_tensors[i].data = init_weights[i]
        else:
            for i in range(len(self._layer_weight_tensors)):
                if use_bias:
                    init_params(self._layer_weight_tensors[i],
                                self._layer_bias_vectors[i])
                else:
                    init_params(self._layer_weight_tensors[i])

        self._is_properly_setup()

    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            weights (list or dict): If a list of parameter tensors is given and
                context modulation is used (see argument ``use_context_mod`` in
                constructor), then these parameters are interpreted as context-
                modulation parameters if the length of ``weights`` equals
                :code:`2*len(net.context_mod_layers)`. Otherwise, the length is
                expected to be equal to the length of the attribute
                :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.

                Alternatively, a dictionary can be passed with the possible
                keywords ``internal_weights`` and ``mod_weights``. Each keyword
                is expected to map onto a list of tensors.
                The keyword ``internal_weights`` refers to all weights of this
                network except for the weights of the context-modulation layers.
                The keyword ``mod_weights``, on the other hand, refers
                specifically to the weights of the context-modulation layers.
                It is not necessary to specify both keywords.
            distilled_params: Will be passed as ``running_mean`` and
                ``running_var`` arguments of method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if
                batch normalization is used.
            condition (optional, int or dict): If ``int`` is provided, then this
                argument will be passed as argument ``stats_id`` to the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if
                batch normalization is used.

                If a ``dict`` is provided instead, the following keywords are
                allowed:

                    - ``bn_stats_id``: Will be handled as ``stats_id`` of the
                      batchnorm layers as described above.
                    - ``cmod_ckpt_id``: Will be passed as argument ``ckpt_id``
                      to the method
                      :meth:`utils.context_mod_layer.ContextModLayer.forward`.

        Returns:
            (tuple): Tuple containing:

            - **y**: The output of the network.
            - **h_y** (optional): If ``out_fn`` was specified in the
              constructor, then this value will be returned. It is the last
              hidden activation (before the ``out_fn`` has been applied).
        """
        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
        n_cm = 0 if self.context_mod_layers is None else \
            2 * len(self.context_mod_layers)

        if weights is None:
            weights = self.weights

            if self._use_context_mod:
                cm_weights = weights[:n_cm]
                int_weights = weights[n_cm:]
            else:
                int_weights = weights
        else:
            int_weights = None
            cm_weights = None

            if isinstance(weights, dict):
                assert('internal_weights' in weights.keys() or \
                       'mod_weights' in weights.keys())
                if 'internal_weights' in weights.keys():
                    int_weights = weights['internal_weights']
                if 'mod_weights' in weights.keys():
                    cm_weights = weights['mod_weights']
            else:
                if self._use_context_mod and \
                        len(weights) == n_cm:
                    cm_weights = weights
                else:
                    assert(len(weights) == len(self.param_shapes))
                    if self._use_context_mod:
                        cm_weights = weights[:n_cm]
                        int_weights = weights[n_cm:]
                    else:
                        int_weights = weights

            if self._use_context_mod and cm_weights is None:
                if self._context_mod_no_weights:
                    raise Exception('Network was generated without weights ' +
                        'for context-mod layers. Hence, they must be passed ' +
                        'via the "weights" option.')
                cm_weights = self.weights[:n_cm]
            if int_weights is None:
                if self._no_weights:
                    raise Exception('Network was generated without internal ' +
                        'weights. Hence, they must be passed via the ' +
                        '"weights" option.')
                if self._context_mod_no_weights:
                    int_weights = self.weights
                else:
                    int_weights = self.weights[n_cm:]

            # Note, context-mod weights might have different shapes, as they
            # may be parametrized on a per-sample basis.
            if self._use_context_mod:
                assert(len(cm_weights) == len(self._context_mod_shapes))
            int_shapes = self.param_shapes[n_cm:]
            assert(len(int_weights) == len(int_shapes))
            for i, s in enumerate(int_shapes):
                assert(np.all(np.equal(s, list(int_weights[i].shape))))

        cm_ind = 0
        bn_ind = 0

        if self._use_batch_norm:
            n_bn = 2 * len(self.batchnorm_layers)
            bn_weights = int_weights[:n_bn]
            layer_weights = int_weights[n_bn:]
        else:
            layer_weights = int_weights

        w_weights = []
        b_weights = []
        for i, p in enumerate(layer_weights):
            if self.has_bias and i % 2 == 1:
                b_weights.append(p)
            else:
                w_weights.append(p)

        ########################
        ### Parse condition ###
        #######################

        bn_cond = None
        cmod_cond = None

        if condition is not None:
            if isinstance(condition, dict):
                assert('bn_stats_id' in condition.keys() or \
                       'cmod_ckpt_id' in condition.keys())
                if 'bn_stats_id' in condition.keys():
                    bn_cond = condition['bn_stats_id']
                if 'cmod_ckpt_id' in condition.keys():
                    cmod_cond = condition['cmod_ckpt_id']
            else:
                bn_cond = condition

        ######################################
        ### Select batchnorm running stats ###
        ######################################
        if self._use_batch_norm:
            nn = len(self._batchnorm_layers)
            running_means = [None] * nn
            running_vars = [None] * nn

        if distilled_params is not None:
            if not self._distill_bn_stats:
                raise ValueError('Argument "distilled_params" can only be ' +
                                 'provided if the return value of ' +
                                 'method "distillation_targets()" is not None.')
            shapes = self.hyper_shapes_distilled
            assert(len(distilled_params) == len(shapes))
            for i, s in enumerate(shapes):
                assert(np.all(np.equal(s, list(distilled_params[i].shape))))

            # Extract batchnorm stats from distilled_params
            for i in range(0, len(distilled_params), 2):
                running_means[i//2] = distilled_params[i]
                running_vars[i//2] = distilled_params[i+1]

        elif self._use_batch_norm and self._bn_track_stats and \
                bn_cond is None:
            for i, bn_layer in enumerate(self._batchnorm_layers):
                running_means[i], running_vars[i] = bn_layer.get_stats()

        ###########################
        ### Forward Computation ###
        ###########################
        hidden = x

        # Context-dependent modulation of inputs directly.
        if self._use_context_mod and self._context_mod_inputs:
            hidden = self._context_mod_layers[cm_ind].forward(hidden,
                weights=cm_weights[2*cm_ind:2*cm_ind+2], ckpt_id=cmod_cond)
            cm_ind += 1

        for l in range(len(w_weights)):
            W = w_weights[l]
            if self.has_bias:
                b = b_weights[l]
            else:
                b = None

            # Linear layer.
            hidden = self._spec_norm(F.linear(hidden, W, bias=b))

            # Only for hidden layers.
            if l < len(w_weights) - 1:
                # Context-dependent modulation (pre-activation).
                if self._use_context_mod and \
                        not self._context_mod_post_activation:
                    hidden = self._context_mod_layers[cm_ind].forward(hidden,
                        weights=cm_weights[2*cm_ind:2*cm_ind+2],
                        ckpt_id=cmod_cond)
                    cm_ind += 1

                # Batch norm
                if self._use_batch_norm:
                    hidden = self._batchnorm_layers[bn_ind].forward(hidden,
                        running_mean=running_means[bn_ind],
                        running_var=running_vars[bn_ind],
                        weight=bn_weights[2*bn_ind],
                        bias=bn_weights[2*bn_ind+1], stats_id=bn_cond)
                    bn_ind += 1

                # Dropout
                if self._dropout_rate != -1:
                    hidden = self._dropout(hidden)

                # Non-linearity
                if self._a_fun is not None:
                    hidden = self._a_fun(hidden)

                # Context-dependent modulation (post-activation).
                if self._use_context_mod and self._context_mod_post_activation:
                    hidden = self._context_mod_layers[cm_ind].forward(hidden,
                        weights=cm_weights[2*cm_ind:2*cm_ind+2],
                        ckpt_id=cmod_cond)
                    cm_ind += 1

        # Context-dependent modulation in output layer.
        if self._use_context_mod and not self._no_last_layer_context_mod:
            hidden = self._context_mod_layers[cm_ind].forward(hidden,
                weights=cm_weights[2*cm_ind:2*cm_ind+2], ckpt_id=cmod_cond)

        if self._out_fn is not None:
            return self._out_fn(hidden), hidden

        return hidden

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This method will return the current batch statistics of all batch
        normalization layers if ``distill_bn_stats`` and ``use_batch_norm``
        was set to ``True`` in the constructor.

        Returns:
            The target tensors corresponding to the shapes specified in
            attribute :attr:`hyper_shapes_distilled`.
        """
        if self.hyper_shapes_distilled is None:
            return None

        ret = []
        for bn_layer in self._batchnorm_layers:
            ret.extend(bn_layer.get_stats())

        return ret

    @staticmethod
    def weight_shapes(n_in=1, n_out=1, hidden_layers=[10, 10], use_bias=True):
        """Compute the tensor shapes of all parameters in a fully-connected
        network.

        Args:
            n_in: Number of inputs.
            n_out: Number of output units.
            hidden_layers: A list of ints, each number denoting the size of a
                hidden layer.
            use_bias: Whether the FC layers should have biases.

        Returns:
            A list of list of integers, denoting the shapes of the individual
            parameter tensors.
        """
        shapes = []

        prev_dim = n_in
        layer_out_sizes = hidden_layers + [n_out]
        for i, size in enumerate(layer_out_sizes):
            shapes.append([size, prev_dim])
            if use_bias:
                shapes.append([size])
            prev_dim = size

        return shapes

if __name__ == '__main__':
    pass



