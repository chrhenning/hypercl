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
- **title**          :utils/batchnorm_layer.py
- **author**         :ch
- **contact**        :henningc@ethz.ch
- **created**        :09/02/2019
- **version**        :1.0
- **python_version** :3.6.8

Implementation of a hypernet compatible batchnorm layer.

The joint use of batch-normalization and hypernetworks is not straight forward,
mainly due to the statistics accumulated by the batch-norm operation which
expect the weights of the main network to only change slowly. If a hypernetwork
replaces the whole set of weights, the statistics previously estimated by the
batch-norm layer might be completely off.

To circumvent this problem, we provide multiple solutions:

    - In a continual learning setting with one set of weights per task, we can
      simply estimate and store statistics per task (hence, the batch-norm
      operation has to be conditioned on the task).
    - The statistics are distilled into the hypernetwork. This would require
      the addition of an extra loss term.
    - The statistics can be treated as parameters that are outputted by the
      hypernetwork. In this case, nothing enforces that these "statistics"
      behave similar to statistics that would result from a running estimate
      (hence, the resulting operation might have nothing in common with batch-
      norm).
    - Always use the statistics estimated on the current batch.

Note, we also provide the option of turning off the statistics, in which case
the statistics will be set to zero mean and unit variance. This is helpful when
interpreting batch-normalization as a general form of gain modulation (i.e.,
just applying a shift and scale to neural activities).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from warnings import warn

class BatchNormLayer(nn.Module):
    r"""Hypernetwork-compatible batch-normalization layer.

    Note, batch normalization performs the following operation

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \
            \gamma + \beta

    This class allows to deviate from this standard implementation in order to
    provide the flexibility required when using hypernetworks. Therefore, we
    slightly change the notation to

    .. math::

        y = \frac{x - m_{\text{stats}}^{(t)}}{\sqrt{v_{\text{stats}}^{(t)} + \
                  \epsilon}} * \gamma^{(t)} + \beta^{(t)}

    We use this notation to highlight that the running statistics
    :math:`m_{\text{stats}}^{(t)}` and :math:`v_{\text{stats}}^{(t)}` are not
    necessarily estimates resulting from mean and variance computation but might
    be learned parameters (e.g., the outputs of a hypernetwork).

    We additionally use the superscript :math:`(t)` to denote that the gain
    :math:`\gamma`, offset :math:`\beta` and statistics may be dynamically
    selected based on some external context information.

    This class provides the possibility to checkpoint statistics
    :math:`m_{\text{stats}}^{(t)}` and :math:`v_{\text{stats}}^{(t)}`, but
    **not** gains and offsets.

    .. note::
        If context-dependent gains :math:`\gamma^{(t)}` and offsets
        :math:`\beta^{(t)}` are required, then they have to be maintained
        externally, e.g., via a task-conditioned hypernetwork (see
        `this paper`_ for an example) and passed to the :meth:`forward` method.

        .. _this paper: https://arxiv.org/abs/1906.00695

    Attributes:
        weights: A list of all internal weights of this layer. If all
            weights are assumed to be generated externally, then this
            attribute will be ``None``.
        param_shapes: A list of list of integers. Each list represents the
            shape of a parameter tensor. Note, this attribute is
            independent of the attribute :attr:`weights`, it always comprises
            the shapes of all weight tensors as if the network would be stand-
            alone (i.e., no weights being passed to the :meth:`forward` method).
            Note, unless ``learnable_stats`` is enabled, the layer statistics
            are not considered here.
        hyper_shapes: A list of list of integers. Each list represents the
            shape of a weight tensor that can be passed to the :meth:`forward`
            method. If all weights are maintained internally, then this
            attribute will be ``None``.
            Specifically, this attribute is controlled by the argument
            ``affine``. If ``affine`` is ``True``, this attribute will be
            ``None``. Otherwise this attribute contains the shape of
            :math:`\gamma` and :math:`\beta`.
        num_stats: The number :math:`T` of internally managed statistics
            :math:`\{(m_{\text{stats}}^{(1)}, v_{\text{stats}}^{(1)}), \dots, \
            (m_{\text{stats}}^{(T)}, v_{\text{stats}}^{(T)}) \}`. This number is
            incremented everytime the method :meth:`checkpoint_stats` is called.
    """
    def __init__(self, num_features, momentum=0.1, affine=True,
                 track_running_stats=True, frozen_stats=False,
                 learnable_stats=False):
        r"""
        Args:
            num_features: See argument ``num_features``, for instance, of class
                :class:`torch.nn.BatchNorm1d`.
            momentum: See argument ``momentum`` of class
                :class:`torch.nn.BatchNorm1d`.
            affine: See argument ``affine`` of class
                :class:`torch.nn.BatchNorm1d`. If set to :code:`False`, the
                input activity will simply be "whitened" according to the
                applied layer statistics (except if gain :math:`\gamma` and
                offset :math:`\beta` are passed to the :meth:`forward` method).

                Note, if ``learnable_stats`` is :code:`False`, then setting
                ``affine`` to :code:`False` results in no learnable weights for
                this layer (running stats might still be updated, but not via
                gradient descent).

                Note, even if this option is ``False``, one may still pass a
                gain :math:`\gamma` and offset :math:`\beta` to the
                :meth:`forward` method.
            track_running_stats: See argument ``track_running_stats`` of class
                :class:`torch.nn.BatchNorm1d`.
            frozen_stats: If ``True``, the layer statistics are frozen at their
                initial values of :math:`\gamma = 1` and :math:`\beta = 0`,
                i.e., layer activity will not be whitened.

                Note, this option requires ``track_running_stats`` to be set to
                ``False``.
            learnable_stats: If ``True``, the layer statistics are initialized
                as learnable parameters (:code:`requires_grad=True`).

                Note, these extra parameters will be maintained internally and
                not added to the :attr:`weights`. Statistics can always be
                maintained externally and passed to the :meth:`forward` method.

                Note, this option requires ``track_running_stats`` to be set to
                ``False``.
        """
        super(BatchNormLayer, self).__init__()

        if learnable_stats:
            # FIXME We need our custom stats computation for this.
            # The running stats updated by `torch.nn.functional.batch_norm` do
            # not allow backpropagation.
            # See here on how they are computed:
            # https://github.com/pytorch/pytorch/blob/96fe2b4ecbbd02143d95f467655a2d697282ac32/aten/src/ATen/native/Normalization.cpp#L137
            raise NotImplementedError('Option "learnable_stats" has not been ' +
                                      'implemented yet!')

        if momentum is None:
            # If one wants to implement this, then please note that the
            # attribute `num_batches_tracked` has to be added. Also, note the
            # extra code for computing the momentum value in the forward method
            # of class `_BatchNorm`:
            # https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#_BatchNorm
            raise NotImplementedError('This reimplementation of PyTorch its ' +
                                      'batchnorm layer does not support ' +
                                      'setting "momentum" to None.')

        if learnable_stats and track_running_stats:
            raise ValueError('Option "track_running_stats" must be set to ' +
                             'False when enabling "learnable_stats".')

        if frozen_stats and track_running_stats:
            raise ValueError('Option "track_running_stats" must be set to ' +
                             'False when enabling "frozen_stats".')

        self._num_features = num_features
        self._momentum = momentum
        self._affine = affine
        self._track_running_stats = track_running_stats
        self._frozen_stats = frozen_stats
        self._learnable_stats = learnable_stats

        self.register_buffer('_num_stats', torch.tensor(0, dtype=torch.long))

        self._weights = nn.ParameterList()
        self._param_shapes = [[num_features], [num_features]]

        if affine:
            # Gamma
            self.register_parameter('scale', nn.Parameter( \
                torch.Tensor(num_features), requires_grad=True))
            # Beta
            self.register_parameter('bias', nn.Parameter( \
                torch.Tensor(num_features), requires_grad=True))

            self._weights.append(self.scale)
            self._weights.append(self.bias)

            nn.init.ones_(self.scale)
            nn.init.zeros_(self.bias)

        elif not learnable_stats:
            self._weights = None

        if learnable_stats:
            # Don't forget to add the new params to `self._weights`.
            # Don't forget to add shapes to `self._param_shapes`.
            raise NotImplementedError()

        elif track_running_stats or frozen_stats:
            # Note, in case of frozen stats, we just don't update the stats
            # initialized here later on.
            self.checkpoint_stats()
        else:
            mname, vname = self._stats_names(0)
            self.register_buffer(mname, None)
            self.register_buffer(vname, None)

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

        Returns:
            A list of lists of integers.
        """
        # FIXME not implemented attribute. Do we even need the attribute, given
        # that all components are individually passed to the forward method?
        raise NotImplementedError('Not implemented yet!')
        return self._hyper_shapes

    @property
    def num_stats(self):
        """Getter for read-only attribute :attr:`num_stats`.

        Returns:
            (int)
        """
        return self._num_stats

    def forward(self, inputs, running_mean=None, running_var=None, weight=None,
                bias=None, stats_id=None):
        r"""Apply batch normalization to given layer activations.

        Based on the state if this module (attribute :attr:`training`), the
        configuration of this layer and the parameters currently passed, the
        behavior of this function will be different.

        The core of this method still relies on the function
        :func:`torch.nn.functional.batch_norm`. In the following we list the
        different behaviors of this method based on the context.

        **In training mode:**

        We first consider the case that this module is in training mode, i.e.,
        :meth:`torch.nn.Module.train` has been called.

        Usually, during training, the running statistics are not used when
        computing the output, instead the statistics computed on the current
        batch are used (denoted by *use batch stats* in the table below).
        However, the batch statistics are typically updated during training
        (denoted by *update running stats* in the table below).

        The above described scenario would correspond to passing batch
        statistics to the function :func:`torch.nn.functional.batch_norm` and
        setting the parameter ``training`` to ``True``.

        +----------------------+---------------------+-------------------------+
        | **training mode**    | **use batch stats** | **update running stats**|
        +----------------------+---------------------+-------------------------+
        | given stats          | Yes                 | Yes                     |
        +----------------------+---------------------+-------------------------+
        | track running stats  | Yes                 | Yes                     |
        +----------------------+---------------------+-------------------------+
        | frozen stats         | No                  | No                      |
        +----------------------+---------------------+-------------------------+
        | learnable stats      | Yes                 | Yes [1]_                |
        +----------------------+---------------------+-------------------------+
        |no track running stats| Yes                 | No                      |
        +----------------------+---------------------+-------------------------+

        The meaning of each row in this table is as follows:

            - **given stats**: External stats are provided via the parameters
              ``running_mean`` and ``running_var``.
            - **track running stats**: If ``track_running_stats`` was set to
              ``True`` in the constructor and no stats were given.
            - **frozen stats**: If ``frozen_stats`` was set to ``True`` in the
              constructor and no stats were given.
            - **learnable stats**: If ``learnable_stats`` was set to ``True`` in
              the constructor and no stats were given.
            - **no track running stats**: If none of the above options apply,
              then the statistics will always be computed from the current batch
              (also in eval mode).

        .. note::
            If provided, running stats specified via ``running_mean`` and
            ``running_var`` always have priority.

        .. [1] We use a custom implementation to update the running statistics,
           that is compatible with backpropagation.

        **In evaluation mode:**

        We now consider the case that this module is in evaluation mode, i.e.,
        :meth:`torch.nn.Module.eval` has been called.

        Here is the same table as above just for the evaluation mode.

        +----------------------+---------------------+-------------------------+
        | **evaluation mode**  | **use batch stats** | **update running stats**|
        +----------------------+---------------------+-------------------------+
        | track running stats  | No                  | No                      |
        +----------------------+---------------------+-------------------------+
        | frozen stats         | No                  | No                      |
        +----------------------+---------------------+-------------------------+
        | learnable stats      | No                  | No                      |
        +----------------------+---------------------+-------------------------+
        | given stats          | No                  | No                      |
        +----------------------+---------------------+-------------------------+
        |no track running stats| Yes                 | No                      |
        +----------------------+---------------------+-------------------------+

        Args:
            inputs: The inputs to the batchnorm layer.
            running_mean (optional): Running mean stats
                :math:`m_{\text{stats}}`. This option has priority, i.e., any
                internally maintained statistics are ignored if given.

                .. note::
                    If specified, then ``running_var`` also has to be specified.
            running_var (optional): Similar to option ``running_mean``, but for
                the running variance stats :math:`v_{\text{stats}}`

                .. note::
                    If specified, then ``running_mean`` also has to be
                    specified.
            weight (optional): The gain factors :math:`\gamma`. If given, any
                internal gains are ignored. If option ``affine`` was set to
                ``False`` in the constructor and this option remains ``None``,
                then no gains are multiplied to the "whitened" inputs.
            bias (optional): The behavior of this option is similar to option
                ``weight``, except that this option represents the offsets
                :math:`\beta`.
            stats_id: This argument is optional except if multiple running
                stats checkpoints exist (i.e., attribute :attr:`num_stats` is
                greater than 1) and no running stats have been provided to this
                method.

                .. note::
                    This argument is ignored if running stats have been passed.

        Returns:
            The layer activation ``inputs`` after batch-norm has been applied.
        """
        assert(running_mean is None and running_var is None or \
               running_mean is not None and running_var is not None)

        if not self._affine:
            if weight is None or bias is None:
                raise ValueError('Layer was generated in non-affine mode. ' +
                                 'Therefore, arguments "weight" and "bias" ' +
                                 'may not be None.')

        # No gains given but we have internal gains.
        # Otherwise, if no gains are given we leave `weight` as None.
        if weight is None and self._affine:
            weight = self.scale
        if bias is None and self._affine:
            bias = self.bias

        stats_given = running_mean is not None

        if (running_mean is None or running_var is None):
            if stats_id is None  and self.num_stats > 1:
                raise ValueError('Parameter "stats_id" is not defined but ' +
                                 'multiple running stats are available.')
            elif self._track_running_stats:
                if stats_id is None:
                    stats_id = 0
                assert(stats_id < self.num_stats)

                rm, rv = self.get_stats(stats_id)

                if running_mean is None:
                    running_mean = rm
                if running_var is None:
                    running_var = rv
        elif stats_id is not None:
            warn('Parameter "stats_id" is ignored since running stats have ' +
                 'been provided.')

        momentum = self._momentum

        if stats_given or self._track_running_stats:
            return F.batch_norm(inputs, running_mean, running_var,
                                weight=weight, bias=bias,
                                training=self.training, momentum=momentum)

        if self._learnable_stats:
            raise NotImplementedError()

        if self._frozen_stats:
            return F.batch_norm(inputs, running_mean, running_var,
                                weight=weight, bias=bias, training=False)

            # TODO implement scale and shift here. Note, that `running_mean` and
            # `running_var` are always 0 and 1, resp. Therefore, the call to
            # `F.batch_norm` is a waste of computation.
            #ret = inputs
            #if weight is not None:
            #    # Multiply `ret` with `weight` such that dimensions are
            #    # respected.
            #    pass
            #if bias is not None:
            #    # Add `bias` to modified `ret` such that dimensions are
            #    # respected.
            #    pass
            #return ret

        else:
            assert(not self._track_running_stats)

            # Always compute statistics based on current batch.
            return F.batch_norm(inputs, None, None, weight=weight, bias=bias,
                                training=True, momentum=momentum)

    def checkpoint_stats(self, device=None):
        """Buffers for a new set of running stats will be registered.

        Calling this function will also increment the attribute
        :attr:`num_stats`.

        Args:
            device (optional): If not provided, the newly created statistics
                will either be moved to the device of the most recent statistics
                or to CPU if no prior statistics exist.
        """
        assert(self._track_running_stats or \
               self._frozen_stats and self._num_stats == 0)

        if device is None:
            if self.num_stats > 0:
                mname_old, _ = self._stats_names(self._num_stats-1)
                device = getattr(self, mname_old).device

        if self._learnable_stats:
            raise NotImplementedError()

        mname, vname = self._stats_names(self._num_stats)
        self._num_stats += 1

        self.register_buffer(mname, torch.zeros(self._num_features,
                                                device=device))
        self.register_buffer(vname, torch.ones(self._num_features,
                                               device=device))

    def get_stats(self, stats_id=None):
        """Get a set of running statistics (means and variances).

        Args:
            stats_id (optional): ID of stats. If not provided, the most recent
                stats are returned.

        Returns:
            (tuple): Tuple containing:

            - **running_mean**
            - **running_var**
        """
        if stats_id is None:
            stats_id = self.num_stats - 1
        assert(stats_id < self.num_stats)

        mname, vname = self._stats_names(stats_id)

        running_mean = getattr(self, mname)
        running_var = getattr(self, vname)

        return running_mean, running_var


    def _stats_names(self, stats_id):
        """Get the buffer names for mean and variance statistics depending on
        the ``stats_id``, i.e., the ID of the stats checkpoint.

        Args:
            stats_id: ID of stats.

        Returns:
            (tuple): Tuple containing:

            - **mean_name**
            - **var_name**
        """
        mean_name = 'mean_%d' % stats_id
        var_name = 'var_%d' % stats_id

        return mean_name, var_name

if __name__ == '__main__':
    pass


