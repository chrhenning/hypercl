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
@title           :toy_regression/hyper_model.py
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
    "no_weights" is True: No spectral normalization is used and the hypernetwork
        can be implemented as shown here.
    "no_weights" is False and spectral normalization is used:
        The hypernetwork needs consist of a set of modules rather than a set of
        parameters of modules (which can then passed to methods from
        nn.functional). E.g., in the constructor of the linear hypernetwork we
        have to create instances of nn.Linear:
            nn.utils.spectral_norm(nn.Linear(n, m))
        Though, if one wants to keep the possibility of easily adding parameters
        (see argument "dTheta" of forward method), one should continue using the
        methods provided in nn.functional and instead find a way to wrap
        parameters inside modules that fulfill no other purpose.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from classifier.convnet import Classifier
from utils.module_wrappers import CLHyperNetInterface
from utils.misc import init_params

class HyperNetwork(nn.Module, CLHyperNetInterface):
    """This network consists of a series of fully-connected layers to generate
    weights for another fully-connected network.

    Attributes (additional to base class):
    """
    def __init__(self, target_shapes, num_tasks, layers=[50, 100], verbose=True,
                 te_dim=8, no_te_embs=False, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_weights=False, init_weights=None,
                 ce_dim=None, dropout_rate=-1, use_spectral_norm=False,
                 use_batch_norm=False, noise_dim=-1):
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
                to the hypernetwork. If this option is None, no additional input
                is expected. Otherwise, an additional embedding has to be passed
                to the forward function.
                A typical usecase would be a chunk embedding.
            dropout_rate (optional): If -1, no dropout will be applied.
                Otherwise a number between 0 and 1 is expected, denoting the
                dropout of hidden layers.
            use_spectral_norm: Enable spectral normalization for all layers.
            use_batch_norm: If True, batchnorm will be applied to all hidden
                layers.
            noise_dim (optional): If -1, no noise will be applied.
                Otherwise the hypernetwork will receive as additional input
                zero-mean Gaussian noise with unit variance during training
                (zeroes will be inputted during eval-mode). Note, if a batch of
                inputs is given, then a different noise vector is generated for
                every sample in the batch.
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

        ### Hidden layers
        self._gen_layers(layers, te_dim, use_bias, no_weights, init_weights,
                         ce_dim, noise_dim)

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

        ntheta = Classifier.num_hyper_weights(self._theta_shapes)
        ntembs = int(np.sum([t.numel() for t in self._task_embs])) \
                if not no_te_embs else 0
        self._num_weights = ntheta + ntembs

        self._num_outputs = Classifier.num_hyper_weights(self.target_shapes)

        if verbose:
            print('Constructed hypernetwork with %d parameters (' % (ntheta \
                  + ntembs) + '%d network weights + %d task embedding weights).'
                  % (ntheta, ntembs))
            print('The hypernetwork has a total of %d outputs.'
                  % self._num_outputs)

        self._is_properly_setup()

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
            outputs.append(W)
            j += 1

        return outputs

if __name__ == '__main__':
    pass
