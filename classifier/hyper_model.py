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
@title           :hyper_model.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :10/25/2018
@version         :1.0
@python_version  :3.6.6

Implements a hypernetwork that generates the weights for the main network
defined in module main_model.py (located in the same directory).
"""
import torch
import torch.nn as nn
import numpy as np
from warnings import warn

from classifier.convnet import Classifier
from toy_example.hyper_model import HyperNetwork
from utils.module_wrappers import CLHyperNetInterface

class ChunkedHyperNetworkHandler(nn.Module, CLHyperNetInterface):
    """This class handles instances of the class HyperNetwork to produce
    the weights of a full main network. I.e., it generates one instance of the
    class HyperNetwork and handles all the embedding vectors. Additionally,
    it provides an easy interface to generate the weights of the main network.

    Attributes (additional to base class):
    """
    def __init__(self, target_shapes, num_tasks, chunk_dim=2586,
                 layers=[50, 100], te_dim=8, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_weights=False, ce_dim=None,
                 init_weights=None, dropout_rate=-1, noise_dim=-1):
        """Initialize the network(s) and all embeddings.

        Note, to implement "noise_dim" this class does not make use of the
        underlying full hypernetwork. Instead, it concatenates noise to the
        chunk embeddings before inputting them to the full hypernet (in this
        way, we make sure that we use the same noise (for all chunks) while
        producing one set of main network weights).

        Note, if "no_weights" is set, then there also won't be internal chunk
        embeddings.

        Args:
            See docstring of class "HyperNetwork" in "../toy_example".
            chunk_dim: The number of weights produced by single hypernet.
            ce_dim: The size if the chunk embeddings.
        """
        # FIXME find a way using super to handle multiple inheritence.
        #super(ChunkedHyperNetworkHandler, self).__init__()
        nn.Module.__init__(self)
        CLHyperNetInterface.__init__(self)

        assert(len(target_shapes) > 0)
        assert (init_weights is None or no_weights is False)
        assert(ce_dim is not None)
        self._target_shapes = target_shapes
        self._num_tasks = num_tasks
        self._chunk_dim = chunk_dim
        self._layers = layers
        self._use_bias = use_bias
        self._act_fn = activation_fn
        self._init_weights = init_weights
        self._no_weights = no_weights
        self._te_dim = te_dim
        self._noise_dim = noise_dim

        # FIXME: weights should incorporate chunk embeddings as they are part of
        # theta.
        if init_weights is not None:
            warn('Argument "init_weights" does not yet allow initialization ' +
                 'of chunk embeddings.')

        ### Generate Hypernet with chunk_dim output.
        self._hypernet = HyperNetwork([[chunk_dim]], num_tasks, verbose=False,
            layers=layers, te_dim=te_dim, activation_fn=activation_fn,
            use_bias=use_bias, no_weights=no_weights, init_weights=init_weights,
            ce_dim=ce_dim + (noise_dim if noise_dim != -1 else 0),
            dropout_rate=dropout_rate, noise_dim=-1)

        self._num_outputs = Classifier.num_hyper_weights(self._target_shapes)
        ### Generate embeddings for all weight chunks.
        self._num_chunks = int(np.ceil(self._num_outputs / chunk_dim))
        if no_weights:
            self._embs = None
        else:
            self._embs = nn.Parameter(data=torch.Tensor(self._num_chunks,
                ce_dim), requires_grad=True)
            nn.init.normal_(self._embs, mean=0., std=1.)

        # Note, the chunk embeddings are part of theta.
        hdims = self._hypernet.theta_shapes
        ntheta = Classifier.num_hyper_weights(hdims) + \
            (self._embs.numel() if not no_weights else 0)

        ntembs = int(np.sum([t.numel() for t in self.get_task_embs()]))
        self._num_weights = ntheta + ntembs
        print('Constructed hypernetwork with %d parameters ' % (ntheta \
              + ntembs) + '(%d network weights + %d task embedding weights).'
              % (ntheta, ntembs))

        print('The hypernetwork has a total of %d outputs.' % self._num_outputs)

        self._theta_shapes = [[self._num_chunks, ce_dim]] + \
            self._hypernet.theta_shapes

        self._is_properly_setup()

    def chunk_embeddings(self):
        """Get the chunk embeddings used to produce a full set of main network
        weights with the underlying (small) hypernetwork.

        Returns:
            A list of all chunk embedding vectors.
        """
        return list(torch.split(self._embs, 1, dim=0))

    # @override from CLHyperNetInterface
    def forward(self, task_id=None, theta=None, dTheta=None, task_emb=None,
                ext_inputs=None, squeeze=True):
        """Implementation of abstract super class method.
        
        Note, this methods can't handle external inputs yet!
        
        The method will iterate through the set of internal chunk embeddings,
        calling the internally maintained (small) full hypernetwork for each,
        in order to generate a full set of main network weights.
        """
        if task_id is None and task_emb is None:
            raise Exception('The hyper network has to get either a task ID' +
                            'to choose the learned embedding or directly ' +
                            'get an embedding as input (e.g. from a task ' +
                            'recognition model).')

        if not self.has_theta and theta is None:
            raise Exception('Network was generated without internal weights. ' +
                            'Hence, "theta" option may not be None.')

        if ext_inputs is not None:
            # FIXME If this will be implemented, please consider:
            # * batch size will have to be multiplied based on num chunk
            #   embeddings and the number of external inputs -> large batches
            # * noise dim must adhere correct behavior (different noise per
            #   external input).
            raise NotImplementedError('This hypernetwork implementation does ' +
                'not yet support the passing of external inputs.')

        if theta is None:
            theta = self.theta
        else:
            assert(len(theta) == len(self.theta_shapes))
            assert(np.all(np.equal(self._embs.shape, list(theta[0].shape))))

        chunk_embs = theta[0]
        hnet_theta = theta[1:]

        if dTheta is not None:
            assert(len(dTheta) == len(self.theta_shapes))

            chunk_embs = chunk_embs + dTheta[0]
            hnet_dTheta = dTheta[1:]
        else:
            hnet_dTheta = None

        # Concatenate the same noise to all chunks, such that it can be
        # viewed as if it were an external input.
        if self._noise_dim != -1:
            if self.training:
                eps = torch.randn((1, self._noise_dim))
            else:
                eps = torch.zeros((1, self._noise_dim))
            if self._embs.is_cuda:
                eps = eps.to(self._embs.get_device())
                
            eps = eps.expand(self._num_chunks, self._noise_dim)
            chunk_embs = torch.cat([chunk_embs, eps], dim=1)

        # get chunked weights from HyperNet
        weights = self._hypernet.forward(task_id=task_id, theta=hnet_theta,
            dTheta=hnet_dTheta, task_emb=task_emb, ext_inputs=chunk_embs)
        weights = weights[0].view(1, -1)

        ### Reshape weights dependent on the main networks architecture.
        ind = 0
        ret = []
        for j, s in enumerate(self.target_shapes):
            num = int(np.prod(s))
            W = weights[0][ind:ind+num]
            ind += num
            ret.append(W.view(*s))

        return ret

    # @override from CLHyperNetInterface
    @property
    def theta(self):
        """Getter for read-only attribute theta.

        Theta are all learnable parameters of the chunked hypernet including
        the chunk embeddings that need to be learned.
        Not included are the task embeddings, i.e., theta comprises
        all parameters that should be regularized in order to avoid
        catastrophic forgetting when training the hypernetwork in a Continual
        Learning setting.
        Note, chunk embeddings are prepended to the list of weights "theta" from
        the internal full hypernetwork.

        Returns:
            A list of tensors or None, if "no_weights" was set to True in
            the constructor of this class.
        """
        return [self._embs] + list(self._hypernet.theta)

    # @override from CLHyperNetInterface
    def get_task_embs(self):
        """Overriden super class method."""
        return self._hypernet.get_task_embs()

    # @override from CLHyperNetInterface
    def get_task_emb(self, task_id):
        """Overriden super class method."""
        return self._hypernet.get_task_emb(task_id)

    # @override from CLHyperNetInterface
    @property
    def has_theta(self):
        """Getter for read-only attribute has_theta."""
        return not self._no_weights

if __name__ == '__main__':
    pass
