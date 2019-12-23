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
- **title**          :mnets/mnet_interface.py
- **author**         :ch
- **contact**        :henningc@ethz.ch
- **created**        :09/20/2019
- **version**        :1.0
- **python_version** :3.6.8

An interface for main networks. The interface ensures that we can consistently
use these networks without knowing their specific implementation.
"""
from abc import ABC, abstractmethod
import numpy as np
from warnings import warn
import torch

class MainNetInterface(ABC):
    """A general interface for main networks, that can be used stand-alone
    (i.e., having their own weights) or with no (or only some) internal
    weights, such that the remaining weights have to be passed through the
    forward function (e.g., they may be generated through a hypernetwork).

    Attributes:
        weights: A list of all internal weights of the main network. If all
            weights are assumed to be generated externally, then this
            attribute will be ``None``.

            Simply speaking, the parameters listed here should be passed to
            the optimizer.
        param_shapes: A list of list of integers. Each list represents the
            shape of a parameter tensor. Note, this attribute is
            independent of the attribute :attr:`weights`, it always comprises
            the shapes of all weight tensors as if the network would be stand-
            alone (i.e., no weights being passed to the :meth:`forward`
            method).
        hyper_shapes: A list of list of integers. Each list represents the
            shape of a weight tensor that has to be passed to the
            :meth:`forward` method. If all weights are maintained internally,
            then this attribute will be ``None``.

            .. deprecated:: 1.0
                This attribute has been renamed to :attr:`hyper_shapes_learned`.
        hyper_shapes_learned: A list of list of integers. Each list represents
            the shape of a weight tensor that has to be passed to the
            :meth:`forward` method during training. If all weights are
            maintained internally, then this attribute will be ``None``.
        hyper_shapes_distilled: A list of list of integers. This attribute is
            complementary to attribute :attr:`hyper_shapes_learned`, which
            contains shapes of tensors that are learned through the
            hypernetwork. In contrast, this attribute should contain the shapes
            of tensors that are not needed by the main network during training
            (as it learns or calculates the tensors itself), but should be
            distilled into a hypernetwork after training in order to avoid
            increasing memory consumption.

            The attribute is ``None`` if no tensors have to be distilled into
            a hypernetwork.

            For instance, if batch normalization is used, then the attribute
            :attr:`hyper_shapes_learned` might contain the batch norm weights
            whereas the attribute :attr:`hyper_shapes_distilled` contains the
            running statistics, which are first estimated by the main network
            during training and later distilled into the hypernetwork.
        has_bias: Whether layers in this network have bias terms.
        has_fc_out: Whether the output layer of the network is a fully-
            connected layer.
        mask_fc_out: If this attribute is set to ``True``, it is implicitly
            assumed that if :attr:`hyper_shapes` is not ``None``, the last two
            entries of :attr:`hyper_shapes` are the weights and biases of the
            final fully-connected layer.

            This attribute is helpful, for instance, in multi-head continual
            learning settings. In case we regularize task-specific main network
            weights, it is important to know which weights are specific for an
            output head (as determined by the weights of the final layer).

            .. note::
                Only applies if attribute :attr:`has_fc_out` is ``True``.
        has_linear_out: Is ``True`` if no nonlinearity is applied in the output
            layer.
        num_params: The total number of weights in the parameter tensors
            described by the attribute :attr:`param_shapes`.
        num_internal_params: The number of internally maintained parameters as
            prescribed by attribute :attr:`weights`.
        layer_weight_tensors: These are the actual weight tensors used in layers
            (e.g., weight matrix in fully-connected layer, kernels in
            convolutional layer, ...).

            This attribute is useful when applying a custom initialization to
            these layers.
        layer_bias_vectors: Similar to attribute :attr:`layer_weight_tensors`
            but for the bias vectors in each layer. List should be empty in case
            :attr:`has_bias` is ``False``.
        batchnorm_layers: A list of instances of class
            :class:`utils.batchnorm_layer.BatchNormLayer` in case batch
            normalization is used in this network.

            .. note::
                We explicitly do not support the usage of PyTorch its batchnorm
                layers as class :class:`utils.batchnorm_layer.BatchNormLayer`
                represents a hypernet compatible wrapper for them.
        context_mod_layers: A list of instances of class
            :class:`utils.context_mod_layer.ContextModLayer` in case these are
            used in this network.
    """
    def __init__(self):
        """Initialize the network.

        Args:

        """
        super(MainNetInterface, self).__init__()

        # The following member variables have to be set by all classes that
        # implement this interface.
        self._weights = None
        self._param_shapes = None
        self._hyper_shapes_learned = None
        self._hyper_shapes_distilled = None
        self._has_bias = None
        self._has_fc_out = None
        self._mask_fc_out = None
        self._has_linear_out = None
        self._layer_weight_tensors = None
        self._layer_bias_vectors = None
        self._batchnorm_layers = None
        self._context_mod_layers = None

        # This will be set automatically based on attribute `_param_shapes`.
        self._num_params = None
        # This will be set automatically based on attribute `_weights`.
        self._num_internal_params = None

        # Deprecated, use `_hyper_shapes_learned` instead.
        self._hyper_shapes = None
        # Deprecated, use `_param_shapes` instead.
        self._all_shapes = None

    def _is_properly_setup(self):
        """This method can be used by classes that implement this interface to
        check whether all required properties have been set."""
        assert(self._param_shapes is not None or self._all_shapes is not None)
        if self._param_shapes is None:
            warn('Private member "_param_shapes" should be specified in each ' +
                 'sublcass that implements this interface, since private ' +
                 'member "_all_shapes" is deprecated.', DeprecationWarning)
            self._param_shapes = self._all_shapes

        if self._hyper_shapes is not None or \
                self._hyper_shapes_learned is not None:
            if self._hyper_shapes_learned is None:
                warn('Private member "_hyper_shapes_learned" should be ' +
                     'specified in each sublcass that implements this ' +
                     'interface, since private member "_hyper_shapes" is ' +
                     'deprecated.', DeprecationWarning)
                self._hyper_shapes_learned = self._hyper_shapes
            # FIXME we should actually assert equality if
            # `_hyper_shapes_learned` was not None.
            self._hyper_shapes = self._hyper_shapes_learned

        assert(self._weights is not None or \
               self._hyper_shapes_learned is not None)

        if self._hyper_shapes_learned is None and \
                self.hyper_shapes_distilled is None:
            # Note, `weights` should only contain trainable weights and not
            # other things like running statistics. Thus, things that are passed
            # to an optimizer.
            assert(len(self._weights) == len(self._param_shapes))

        assert(isinstance(self._has_bias, bool))
        assert(isinstance(self._has_fc_out, bool))
        assert(isinstance(self._mask_fc_out, bool))
        assert(isinstance(self._has_linear_out, bool))

        assert(self._layer_weight_tensors is not None)
        assert(self._layer_bias_vectors is not None)
        if self._has_bias:
            assert(len(self._layer_weight_tensors) == \
                   len(self._layer_bias_vectors))

    @property
    def weights(self):
        """Getter for read-only attribute :attr:`weights`.

        Returns:
            A :class:`torch.nn.ParameterList` or ``None``, if no parameters are
            internally maintained.
        """
        return self._weights

    @property
    def param_shapes(self):
        """Getter for read-only attribute :attr:`param_shapes`.

        Returns:
            A list of lists of integers.
        """
        return self._param_shapes

    @property
    def hyper_shapes(self):
        """Getter for read-only attribute :attr:`hyper_shapes`.

        .. deprecated:: 1.0
            This attribute has been renamed to :attr:`hyper_shapes_learned`.

        Returns:
            A list of lists of integers.
        """
        warn('Use atrtibute "hyper_shapes_learned" instead.',
             DeprecationWarning)

        return self.hyper_shapes_learned

    @property
    def hyper_shapes_learned(self):
        """Getter for read-only attribute :attr:`hyper_shapes_learned`.

        Returns:
            A list of lists of integers.
        """
        return self._hyper_shapes_learned

    @property
    def hyper_shapes_distilled(self):
        """Getter for read-only attribute :attr:`hyper_shapes_distilled`.

        Returns:
            A list of lists of integers.
        """
        return self._hyper_shapes_distilled

    @property
    def has_bias(self):
        """Getter for read-only attribute :attr:`has_bias`."""
        return self._has_bias

    @property
    def has_fc_out(self):
        """Getter for read-only attribute :attr:`has_fc_out`."""
        return self._has_fc_out

    @property
    def mask_fc_out(self):
        """Getter for read-only attribute :attr:`mask_fc_out`."""
        return self._mask_fc_out

    @property
    def has_linear_out(self):
        """Getter for read-only attribute :attr:`has_linear_out`."""
        return self._has_linear_out

    @property
    def num_params(self):
        """Getter for read-only attribute :attr:`num_params`.

        Returns:
            (int): Total number of parameters in the network.
        """
        if self._num_params is None:
            self._num_params = int(np.sum([np.prod(l) for l in
                                           self.param_shapes]))
        return self._num_params

    @property
    def num_internal_params(self):
        """Getter for read-only attribute :attr:`num_internal_params`.

        Returns:
            (int): Total number of parameters currently maintained by this
            network instance.
        """
        if self._num_internal_params is None:
            if self.weights is None:
                self._num_internal_params = 0
            else:
                # FIXME should we distinguish between trainable and
                # non-trainable parameters (`p.requires_grad`)?
                self._num_internal_params = int(sum(p.numel() for p in \
                                                    self.weights))
        return self._num_internal_params

    @property
    def layer_weight_tensors(self):
        """Getter for read-only attribute :attr:`layer_weight_tensors`.

        Returns:
            A list (e.g., an instance of class :class:`torch.nn.ParameterList`).
        """
        return self._layer_weight_tensors

    @property
    def layer_bias_vectors(self):
        """Getter for read-only attribute :attr:`layer_bias_vectors`.

        Returns:
            A list (e.g., an instance of class :class:`torch.nn.ParameterList`).
        """
        return self._layer_bias_vectors

    @property
    def batchnorm_layers(self):
        """Getter for read-only attribute :attr:`batchnorm_layers`.

        Returns:
            (:class:`torch.nn.ModuleList`): A list of
            :class:`utils.batchnorm_layer.BatchNormLayer` instances, if batch
            normalization is used.
        """
        return self._batchnorm_layers

    @property
    def context_mod_layers(self):
        """Getter for read-only attribute :attr:`context_mod_layers`.

        Returns:
            (:class:`torch.nn.ModuleList`): A list of
            :class:`utils.context_mod_layer.ContextModLayer` instances, if these
            layers are in use.
        """
        return self._context_mod_layers

    @abstractmethod
    def distillation_targets(self):
        """Targets to be distilled after training.

        If :attr:`hyper_shapes_distilled` is not ``None``, then this method
        can be used to retrieve the targets that should be distilled into an
        external hypernetwork after training.

        The shapes of the returned tensors have to match the shapes specified in
        :attr:`hyper_shapes_distilled`.

        Example:

            Assume a continual learning scenario with a main network that uses
            batch normalization (and tracks running statistics). Then this
            method should be called right after training on a task in order to
            retrieve the running statistics, such that they can be distilled
            into a hypernetwork.

        Returns:
            The target tensors corresponding to the shapes specified in
            attribute :attr:`hyper_shapes_distilled`.
        """
        raise NotImplementedError('TODO implement function')

    @abstractmethod
    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.

        Args:
            x: The inputs :math:`x` to the network.
            weights (optional): List of weight tensors, that are used as network
                parameters. If attribute :attr:`hyper_shapes_learned` is not
                ``None``, then this argument is non-optional and the shapes
                of the weight tensors have to be as specified by
                :attr:`hyper_shapes_learned`.

                Otherwise, this option might still be set but the weight tensors
                must follow the shapes specified by attribute
                :attr:`param_shapes`.
            distilled_params (optional): May only be passed if attribute
                :attr:`hyper_shapes_distilled` is not ``None``.

                If not passed but the network relies on those parameters
                (e.g., batchnorm running statistics), then this method simply
                chooses the current internal representation of these parameters
                as returned by :meth:`distillation_targets`.
            condition (optional): Sometimes, the network will have to be
                conditioned on contextual information, which can be passed via
                this argument and depends on the actual implementation of this
                interface.

                For instance, when using batch normalization in a continual
                learning scenario, where running statistics have been
                checkpointed for every task, then this ``condition`` might be
                the actual task ID, that is passed as the argument ``stats_id``
                of the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward`.

        Returns:
            The output :math:`y` of the network.
        """
        raise NotImplementedError('TODO implement function')

    @staticmethod
    def shapes_to_num_weights(dims):
        """The number of parameters contained in a list of tensors with the
        given shapes.

        Args:
            dims: List of tensor shapes. For instance, the attribute
                :attr:`hyper_shapes_learned`.

        Returns:
            (int)
        """
        return np.sum([np.prod(l) for l in dims])

    def custom_init(self, normal_init=False, normal_std=0.02, zero_bias=True):
        """Initialize weight tensors in attribute :attr:`layer_weight_tensors`
        using Xavier initialization and set bias vectors to 0.

        Note:
            This method will override the default initialization of the network,
            which is often based on :func:`torch.nn.init.kaiming_uniform_`
            for weight tensors (i.e., attribute :attr:`layer_weight_tensors`)
            and a uniform init based on fan-in/fan-out for bias vectors
            (i.e., attribute :attr:`layer_bias_vectors`).

        Args:
            normal_init (bool): Use normal initialization rather than Xavier.
            normal_std (float): The standard deviation when choosing
                ``normal_init``.
            zero_bias (bool): Whether bias vectors should be initialized to
                zero. If ``False``, then bias vectors are left untouched.
        """
        for w in self.layer_weight_tensors:
            if normal_init:
                torch.nn.init.normal_(w, mean=0, std=normal_std)
            else:
                torch.nn.init.xavier_uniform_(w)

        if zero_bias:
            for b in self.layer_bias_vectors:
                torch.nn.init.constant_(b, 0)

if __name__ == '__main__':
    pass


