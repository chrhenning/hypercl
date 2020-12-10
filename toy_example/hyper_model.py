#!/usr/bin/env python3
# Copyright 2018 Christian Henning
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
@title           :toy_example/hyper_model.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :10/23/2018
@version         :1.0
@python_version  :3.6.6

Implementation of a hypernet, that outputs weights for a main network.

This hypernetwork is implemented such that it can be cascaded (i.e., a
hypernetwork can produce the weights of another instance of a hypernetwork).
Note, if one wants to enrich this implementation by adding spectral
normalization (which can only be used for a hypernetwork that has trainable
weights), one would have to prodvide two distinct implementations:

    - `no_weights` is True: No spectral normalization is used and the
      hypernetwork can be implemented as shown here.
    - "no_weights" is False and spectral normalization is used:
      The hypernetwork needs consist of a set of modules rather than a set of
      parameters of modules (which can then passed to methods from
      nn.functional). E.g., in the constructor of the linear hypernetwork we
      have to create instances of nn.Linear:
      :code:`nn.utils.spectral_norm(nn.Linear(n, m))`
      Though, if one wants to keep the possibility of easily adding parameters
      (see argument "dTheta" of forward method), one should continue using the
      methods provided in nn.functional and instead find a way to wrap
      parameters inside modules that fulfill no other purpose.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mnets.mnet_interface import MainNetInterface
from utils import init_utils as iutils
from utils.module_wrappers import CLHyperNetInterface
from utils.torch_utils import init_params

class HyperNetwork(nn.Module, CLHyperNetInterface):
    """This network consists of a series of fully-connected layers to generate
    weights for another fully-connected network.

    Attributes:
        feedback_matrix: A random feedback matrix, that can be used for
            exploration of alternative credit assignment algorithms (such as
            Feedback Alignment).

            .. warning::
                The feedback matrix is stored in this class as a temporary
                solution. In the future it will be moved somewhere else.
    """
    def __init__(self, target_shapes, num_tasks, layers=[50, 100], verbose=True,
                 te_dim=8, no_te_embs=False, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_weights=False, init_weights=None,
                 ce_dim=None, dropout_rate=-1, use_spectral_norm=False,
                 create_feedback_matrix=False, target_net_out_dim=10,
                 random_scale_feedback_matrix = 1.,
                 use_batch_norm=False, noise_dim=-1, temb_std=-1):
        """Build the network. The network will consist of several hidden layers
        and a dedicated output layer for each weight matrix/bias vector.

        The input to the network will be a learned task embedding.

        Args:
            target_shapes: A list of list of integers, denoting the shape of
                each parameter tensor in the main network (hence, determining
                the output of this network).
            num_tasks: The number of task embeddings needed.
            layers: A list of integers, each indicating the size of a hidden
                    layer in this network.
            te_dim: The dimensionality of the task embeddings.
            no_te_embs: If this option is True, no class internal task
                embeddings are constructed and are instead expected to be
                provided to the forward method.
            activation_fn: The nonlinearity used in hidden layers. If None, no
                nonlinearity will be applied.
            use_bias: Whether layers may have bias terms.
            no_weights: If set to True, no trainable parameters will be
                constructed, i.e., weights are assumed to be produced ad-hoc
                by a hypernetwork and passed to the forward function.
                Does not affect task embeddings.
            init_weights (optional): This option is for convenience reasons.
                The option expects a list of parameter values that are used to
                initialize the network weights. As such, it provides a
                convenient way of initializing a network with, for instance, a
                weight draw produced by the hypernetwork.
                Does not affect task embeddings.
            ce_dim (optional): The dimensionality of any additional embeddings,
                (in addition to the task embedding) that will be used as input
                to the hypernetwork. If this option is ``None``, no additional
                input is expected. Otherwise, an additional embedding has to be
                passed to the :meth:`forward` method (see argument
                ``ext_inputs``).
                A typical usecase would be a chunk embedding.
            dropout_rate (optional): If -1, no dropout will be applied.
                Otherwise a number between 0 and 1 is expected, denoting the
                dropout of hidden layers.
            use_spectral_norm: Enable spectral normalization for all layers.
            create_feedback_matrix: A feedback matrix for credit assignment in
                the main network will be created. See attribute
                :attr:`feedback_matrix`.
            target_net_out_dim: Target network output dimension. We need this
                information to create feedback matrices that can be used in
                conjunction with Direct Feedback Alignment. Only needs to be
                specified when enabling ``create_feedback_matrix``.
            random_scale_feedback_matrix: Scale of uniform distribution used
                to create the feedback matrix. Only needs to be specified when
                enabling ``create_feedback_matrix``.
            use_batch_norm: If True, batchnorm will be applied to all hidden
                layers.
            noise_dim (optional): If -1, no noise will be applied.
                Otherwise the hypernetwork will receive as additional input
                zero-mean Gaussian noise with unit variance during training
                (zeroes will be inputted during eval-mode). Note, if a batch of
                inputs is given, then a different noise vector is generated for
                every sample in the batch.
            temb_std (optional): If not -1, the task embeddings will be
                perturbed by zero-mean Gaussian noise with the given std
                (additive noise). The perturbation is only applied if the
                network is in training mode. Note, per batch of external inputs,
                the perturbation of the task embedding will be shared.
        """
        # FIXME find a way using super to handle multiple inheritence.
        #super(HyperNetwork, self).__init__()
        nn.Module.__init__(self)
        CLHyperNetInterface.__init__(self)

        if use_spectral_norm:
            raise NotImplementedError('Spectral normalization not yet ' +
                                      'implemented for this hypernetwork type.')
        if use_batch_norm:
            # Note, batch normalization only makes sense when batch processing
            # is applied during training (i.e., batch size > 1).
            # As long as we only support processing of 1 task embedding, that
            # means that external inputs are required.
            if ce_dim is None:
                raise ValueError('Can\'t use batchnorm as long as ' +
                                 'hypernetwork process more than 1 sample ' +
                                 '("ce_dim" must be specified).')
            raise NotImplementedError('Batch normalization not yet ' +
                                      'implemented for this hypernetwork type.')

        assert(len(target_shapes) > 0)
        assert(no_te_embs or num_tasks > 0)
        self._num_tasks = num_tasks

        assert(init_weights is None or no_weights is False)
        self._no_weights = no_weights
        self._no_te_embs = no_te_embs
        self._te_dim = te_dim
        self._size_ext_input = ce_dim
        self._layers = layers
        self._target_shapes = target_shapes
        self._use_bias = use_bias
        self._act_fn = activation_fn
        self._init_weights = init_weights
        self._dropout_rate = dropout_rate
        self._noise_dim = noise_dim
        self._temb_std = temb_std
        self._shifts = None # FIXME temporary test.

        ### Hidden layers
        self._gen_layers(layers, te_dim, use_bias, no_weights, init_weights,
                         ce_dim, noise_dim)

        if create_feedback_matrix:
            self._create_feedback_matrix(target_shapes, target_net_out_dim,
                                         random_scale_feedback_matrix)

        self._dropout = None
        if dropout_rate != -1:
            assert(dropout_rate >= 0 and dropout_rate <= 1)
            self._dropout = nn.Dropout(dropout_rate)

        # Task embeddings.
        if no_te_embs:
            self._task_embs = None
        else:
            self._task_embs = nn.ParameterList()
            for _ in range(num_tasks):
                self._task_embs.append(nn.Parameter(data=torch.Tensor(te_dim),
                                                    requires_grad=True))
                torch.nn.init.normal_(self._task_embs[-1], mean=0., std=1.)

        self._theta_shapes = self._hidden_dims + self._out_dims

        ntheta = MainNetInterface.shapes_to_num_weights(self._theta_shapes)
        ntembs = int(np.sum([t.numel() for t in self._task_embs])) \
                if not no_te_embs else 0
        self._num_weights = ntheta + ntembs

        self._num_outputs = MainNetInterface.shapes_to_num_weights( \
            self.target_shapes)

        if verbose:
            print('Constructed hypernetwork with %d parameters (' % (ntheta \
                  + ntembs) + '%d network weights + %d task embedding weights).'
                  % (ntheta, ntembs))
            print('The hypernetwork has a total of %d outputs.'
                  % self._num_outputs)

        self._is_properly_setup()

    def _create_feedback_matrix(self, target_shapes, target_net_out_dim, 
                                random_scale_feedback_matrix):
        """Create a feeback matrix for credit assignment as an alternative
        to backprop.

        The matrix will be of dimension:
        ``[target network output dim x hypernetwork output]``.

        Note:
            For now, this method only generates a feedback matrix appropriate
            for Direct Feedback Alignment.

        Args:
            (....): See constructor docstring.
        """
        s = random_scale_feedback_matrix
        self._feedback_matrix = []
        for k in target_shapes:
            dims =  [target_net_out_dim] + k    
            self._feedback_matrix.append(torch.empty(dims).uniform_(-s, s))

    @property
    def feedback_matrix(self):
        """Getter for read-only attribute :attr:`feedback_matrix`.

        The matrix will be of dimension:
        ``[target network output dim x hypernetwork output]``.

        Return:
            (list): Feeback matrix for credit assignment, which is represented
            as a list of torch tensors.
        """

        return self._feedback_matrix

    def _gen_layers(self, layers, te_dim, use_bias, no_weights, init_weights,
                    ce_dim, noise_dim):
        """Generate all layers of this network. This method will create
        the parameters of each layer. Note, this method should only be
        called by the constructor.

        This method will add the attributes "_hidden_dims" and "_out_dims".
        If "no_weights" is False, it will also create an attribute "_weights"
        and initialize all parameters. Otherwise, _weights" is set to None.

        Args:
            See constructur arguments.
        """
        ### Compute the shapes of all parameters.
        # Hidden layers.
        self._hidden_dims = []
        prev_dim = te_dim
        if ce_dim is not None:
            prev_dim += ce_dim
        if noise_dim != -1:
            prev_dim += noise_dim
        for i, size in enumerate(layers):
            self._hidden_dims.append([size, prev_dim])
            if use_bias:
                self._hidden_dims.append([size])
            prev_dim = size
        self._last_hidden_size = prev_dim

        # Output layers.
        self._out_dims = []
        for i, dims in enumerate(self.target_shapes):
            nouts = np.prod(dims)
            self._out_dims.append([nouts, self._last_hidden_size])
            if use_bias:
                self._out_dims.append([nouts])
        if no_weights:
            self._theta = None
            return

        ### Create parameter tensors.
        # If "use_bias" is True, then each odd entry of this list will contain
        # a weight matrix and each even entry a bias vector. Otherwise,
        # it only contains a weight matrix per layer.
        self._theta = nn.ParameterList()
        for i, dims in enumerate(self._hidden_dims + self._out_dims):
            self._theta.append(nn.Parameter(torch.Tensor(*dims),
                                            requires_grad=True))

        if init_weights is not None:
            assert (len(init_weights) == len(self._theta))
            for i in range(len(init_weights)):
                assert (np.all(np.equal(list(init_weights[i].shape),
                                        list(self._theta[i].shape))))
                self._theta[i].data = init_weights[i]
        else:
            for i in range(0, len(self._theta), 2 if use_bias else 1):
                if use_bias:
                    init_params(self._theta[i], self._theta[i + 1])
                else:
                    init_params(self._theta[i])

    # @override from CLHyperNetInterface
    def forward(self, task_id=None, theta=None, dTheta=None, task_emb=None,
                ext_inputs=None, squeeze=True):
        """Implementation of abstract super class method."""
        if task_id is None and task_emb is None:
            raise Exception('The hyper network has to get either a task ID' +
                            'to choose the learned embedding or directly ' +
                            'get an embedding as input (e.g. from a task ' +
                            'recognition model).')

        if not self.has_theta and theta is None:
            raise Exception('Network was generated without internal weights. ' +
                            'Hence, "theta" option may not be None.')

        if theta is None:
            theta = self.theta
        else:
            assert(len(theta) == len(self.theta_shapes))
            for i, s in enumerate(self.theta_shapes):
                assert(np.all(np.equal(s, list(theta[i].shape))))

        if dTheta is not None:
            assert(len(dTheta) == len(self.theta_shapes))

            weights = []
            for i, t in enumerate(theta):
                weights.append(t + dTheta[i])
        else:
            weights = theta

        # Select task embeddings.
        if not self.has_task_embs and task_emb is None:
            raise Exception('The network was created with no internal task ' +
                            'embeddings, thus parameter "task_emb" has to ' +
                            'be specified.')

        if task_emb is None:
            task_emb = self._task_embs[task_id]
        if self.training and self._temb_std != -1:
            task_emb.add(torch.randn_like(task_emb) * self._temb_std)

        # Concatenate additional embeddings to task embedding, if given.
        if self.requires_ext_input and ext_inputs is None:
            raise Exception('The network was created to expect additional ' +
                            'inputs, thus parameter "ext_inputs" has to ' +
                            'be specified.')
        elif not self.requires_ext_input and ext_inputs is not None:
            raise Exception('The network was created to not expect ' +
                            'additional embeddings, thus parameter ' +
                            '"ext_inputs" cannot be specified.')

        if ext_inputs is not None:
            # FIXME at the moment, we only process one task embedding at a time,
            # thus additional embeddings define the batch size.
            batch_size = ext_inputs.shape[0]
            task_emb = task_emb.expand(batch_size, self._te_dim)
            h = torch.cat([task_emb, ext_inputs], dim=1)
        else:
            batch_size = 1
            h = task_emb.expand(batch_size, self._te_dim)

        if self._noise_dim != -1:
            if self.training:
                eps = torch.randn((batch_size, self._noise_dim))
            else:
                eps = torch.zeros((batch_size, self._noise_dim))
            if h.is_cuda:
                eps = eps.to(h.get_device())
            h = torch.cat([h, eps], dim=1)

        # Hidden activations.
        for i in range(0, len(self._hidden_dims), 2 if self._use_bias else 1):
            b = None
            if self._use_bias:
                b = weights[i+1]
            h = F.linear(h, weights[i], bias=b)
            if self._act_fn is not None:
                h = self._act_fn(h)
            if self._dropout is not None:
                h = self._dropout(h)
        outputs = []
        j = 0
        for i in range(len(self._hidden_dims), len(self._theta_shapes),
                       2 if self._use_bias else 1):
            b = None
            if self._use_bias:
                b = weights[i+1]
            W = F.linear(h, weights[i], bias=b)
            W = W.view(batch_size, *self.target_shapes[j])
            if squeeze:
                W = torch.squeeze(W, dim=0)
            if self._shifts is not None: # FIXME temporary test!
                W += self._shifts[j]
            outputs.append(W)
            j += 1

        return outputs

    def apply_hyperfan_init(self, method='in', use_xavier=False,
                            temb_var=1., ext_inp_var=1.):
        r"""Initialize the network using hyperfan init.

        Hyperfan initialization was developed in the following paper for this
        kind of hypernetwork

            "Principled Weight Initialization for Hypernetworks"
            https://openreview.net/forum?id=H1lma24tPB

        The initialization is based on the following idea: When the main network
        would be initialized using Xavier or Kaiming init, then variance of
        activations (fan-in) or gradients (fan-out) would be preserved by using
        a proper variance for the initial weight distribution (assuming certain
        assumptions hold at initialization, which are different for Xavier and
        Kaiming).

        When using these kind of initializations in the hypernetwork, then the
        variance of the initial main net weight distribution would simply equal
        the variance of the input embeddings (which can lead to exploding
        activations, e.g., for fan-in inits).

        The above mentioned paper proposes a quick fix for the type of hypernets
        which have a separate output head per weight tensor in the main network
        (which is the case for this hypernetwork class).

        Assuming that input embeddings are initialized with a certain variance
        (e.g., 1) and we use Xavier or Kaiming init for the hypernet, then the
        variance of the last hidden activation will also be 1.

        Then, we can modify the variance of the weights of each output head in
        the hypernet to obtain the variance for the main net weight tensors that
        we would typically obtain when applying Xavier or Kaiming to the main
        network directly.

        Warning:
            This method currently assumes that 1D target tensors (cmp.
            constructor argument ``target_shapes``) are bias vectors in the
            main network.

        Warning:
            To compute the hyperfan-out initialization of bias vectors, we need
            access to the fan-in of the layer, which we can only compute based
            on the corresponding weight tensor in the same layer. Since there is
            no clean way of matching a bias shape to its corresponging weight
            tensor shape we use the following heuristic, which should be correct
            for most main networks. We assume that the shape directly preceding
            a bias shape in the constructor argument ``target_shapes`` is the
            corresponding weight tensor.

        **Variance of the hypernet input**

        In general, the input to the hypernetwork can be a concatenation of
        multiple embeddings (see description of arguments ``temb_var`` and
        ``ext_inp_var``).

        Let's denote the complete hypernetwork input by
        :math:`\mathbf{x} \in \mathbb{R}^n`, which consists of a task embedding
        :math:`\mathbf{e} \in \mathbb{R}^{n_e}` and an external input
        :math:`\mathbf{c} \in \mathbb{R}^{n_c}`, i.e.,

        .. math::

            \mathbf{x} = \begin{bmatrix} \
            \mathbf{e} \\ \
            \mathbf{c} \
            \end{bmatrix}

        We simply define the variance of an input :math:`\text{Var}(x_j)` as
        the weighted average of the individual variances, i.e.,

        .. math::

            \text{Var}(x_j) \equiv \frac{n_e}{n_e+n_c} \text{Var}(e) + \
                \frac{n_c}{n_e+n_c} \text{Var}(c)

        To see that this is correct, consider a linear layer
        :math:`\mathbf{y} = W \mathbf{x}` or

        .. math::

            y_i &= \sum_j w_{ij} x_j \\ \
                &= \sum_{j=1}^{n_e} w_{ij} e_j + \
                   \sum_{j=n_e+1}^{n_e+n_c} w_{ij} c_{j-n_e}

        Hence, we can compute the variance of :math:`y_i` as follows (assuming
        the typical Xavier assumptions):

        .. math::

            \text{Var}(y) &= n_e \text{Var}(w) \text{Var}(e) + \
                             n_c \text{Var}(w) \text{Var}(c) \\ \
                          &= \frac{n_e}{n_e+n_c} \text{Var}(e) + \
                             \frac{n_c}{n_e+n_c} \text{Var}(c)

        Note, that Xavier would have initialized :math:`W` using
        :math:`\text{Var}(w) = \frac{1}{n} = \frac{1}{n_e+n_c}`.

        Note:
            This method will automatically incorporate the noise embedding that
            is inputted into the network if constructor argument ``noise_dim``
            was set.

        Note:
            All hypernet inputs should be zero mean.

        Args:
            method (str): The type of initialization that should be applied.
                Possible options are:

                - ``in``: Use `Hyperfan-in`.
                - ``out``: Use `Hyperfan-out`.
                - ``harmonic``: Use the harmonic mean of the `Hyperfan-in` and
                  `Hyperfan-out` init.
            use_xavier (bool): Whether Kaiming (``False``) or Xavier (``True``)
                init should be used.
            temb_var (float): The initial variance of the task embeddings.

                .. note::
                    If ``temb_std`` was set in the constructor, then this method
                    will automatically correct the provided ``temb_var`` as 
                    follows: :code:`temb_var += temb_std**2`.
            ext_inp_var (float): The initial variance of the external input.
                Only needs to be specified if external inputs are provided
                (see argument ``ce_dim`` of constructor).
        """
        # FIXME If the network has external inputs and task embeddings, then
        # both these inputs might have different variances. Thus, a single
        # parameter `input_variance` might not be sufficient.
        # Now, we assume that the user provides a proper variance. We could
        # simplify the job for him by providing multiple arguments and compute
        # the weighting ourselves.

        # FIXME Handle constructor arguments `noise_dim` and `temb_std`.
        # Note, we would jost need to add `temb_std**2` to the variance of
        # task embeddings, since the variance of a sum of uncorrelated RVs is
        # just the sum of the individual variances.

        if method not in ['in', 'out', 'harmonic']:
            raise ValueError('Invalid value for argument "method".')
        if not self.has_theta:
            raise ValueError('Hypernet without internal weights can\'t be ' +
                             'initialized.')

        ### Compute input variance ###
        if self._temb_std != -1:
            # Sum of uncorrelated variables.
            temb_var += self._temb_std**2

        assert self._size_ext_input is None or self._size_ext_input > 0
        assert self._noise_dim == -1 or self._noise_dim > 0

        inp_dim = self._te_dim + \
            (self._size_ext_input if self._size_ext_input is not None else 0) \
            + (self._noise_dim if self._noise_dim != -1 else 0)

        input_variance = (self._te_dim / inp_dim) * temb_var
        if self._size_ext_input is not None:
            input_variance += (self._size_ext_input / inp_dim) * ext_inp_var
        if self._noise_dim != -1:
            input_variance += (self._noise_dim / inp_dim) * 1.

        ### Initialize hidden layers to preserve variance ###
        # We initialize biases with 0 (see Xavier assumption 4 in the Hyperfan
        # paper). Otherwise, we couldn't ignore the biases when computing the
        # output variance of a layer.
        # Note, we have to use fan-in init for the hidden layer to ensure the
        # property, that we preserve the input variance.

        for i in range(0, len(self._hidden_dims), 2 if self._use_bias else 1):
            #W = self.theta[i]
            if use_xavier:
                iutils.xavier_fan_in_(self.theta[i])
            else:
                torch.nn.init.kaiming_uniform_(self.theta[i], mode='fan_in',
                                               nonlinearity='relu')

            if self._use_bias:
                #b = self.theta[i+1]
                torch.nn.init.constant_(self.theta[i+1], 0)

        ### Initialize output heads ###
        c_relu = 1 if use_xavier else 2
        # FIXME Not a proper way to figure out whether the hnet produces
        # bias vectors in the mnet.
        c_bias = 1
        for s in self.target_shapes:
            if len(s) == 1:
                c_bias = 2
                break
        # This is how we should do it instead.
        #c_bias = 2 if mnet.has_bias else 1

        j = 0
        for i in range(len(self._hidden_dims), len(self._theta_shapes),
                       2 if self._use_bias else 1):

            # All output heads are linear layers. The biases of these linear
            # layers (called gamma and beta in the paper) are simply initialized
            # to zero.
            if self._use_bias:
                #b = self.theta[i+1]
                torch.nn.init.constant_(self.theta[i+1], 0)

            # We are not interested in the fan-out, since the fan-out is just
            # the number of elements in the corresponding main network tensor.
            # `fan-in` is called `d_k` in the paper and is just the size of the
            # last hidden layer.
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out( \
                self.theta[i])

            out_shape = self.target_shapes[j]

            # FIXME 1D output tensors don't need to be bias vectors. They can
            # be arbitrary embeddings or, for instance, batchnorm weights.
            if len(out_shape) == 1: # Assume output is bias vector.
                m_fan_out = out_shape[0]

                # NOTE For the hyperfan-out init, we also need to know the
                # fan-in of the layer.
                # FIXME We have no proper way at the moment to get the correct
                # fan-in of the layer this bias vector belongs to.
                if j > 0 and len(self.target_shapes[j-1]) > 1:
                    m_fan_in, _ = iutils.calc_fan_in_and_out( \
                        self.target_shapes[j-1])
                else:
                    # FIXME Quick-fix.
                    m_fan_in = m_fan_out

                var_in = c_relu / (2. * fan_in * input_variance)
                num = c_relu * (1. - m_fan_in/m_fan_out)
                denom = fan_in * input_variance
                var_out = max(0, num / denom)

            else:
                m_fan_in, m_fan_out = iutils.calc_fan_in_and_out(out_shape)

                var_in = c_relu / (c_bias * m_fan_in * fan_in * input_variance)
                var_out = c_relu / (m_fan_out * fan_in * input_variance)

            if method == 'in':
                var = var_in
            elif method == 'out':
                var = var_out
            elif method == 'harmonic':
                var = 2 * (1./var_in + 1./var_out)
            else:
                raise ValueError('Method %s invalid.' % method)

            # Initialize output head weight tensor using `var`.
            std = math.sqrt(var)
            a = math.sqrt(3.0) * std
            torch.nn.init._no_grad_uniform_(self.theta[i], -a, a)
            
            j += 1

if __name__ == '__main__':
    pass
