#!/usr/bin/env python3
# Copyright 2019 Johannes von Oswald
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
@title           :mlps.py
@author          :jvo
@contact         :oswald@ini.ethz.ch
@created         :04/11/2019
@version         :1.0
@python_version  :3.6.5

This module contains a classic mlp-like
network to classify either MNIST or CIFAR-10 images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from classifier.convnet import Classifier
from utils.misc import init_params

class MLP(Classifier):
    """The network consists of a given number of fully-
    connected layers. See implementation for details.

    Attributes (additional to base class):
    """
    def __init__(self, layers = [400, 400],
                 in_shape=[784], num_classes=10, verbose=True,
                 no_weights=False, num_heads=1, init_weights=None):
        """Initialize the network.

        Args:
            layers: The architecture of the network defined by the
                dimensions of the hidden layers.
            in_shape: The shape of an input sample. The chosen architecture
                will be adopted accordingly.
            num_classes: The number of output neurons. The chosen architecture
                will be adopted accordingly.
            verbose: Allow printing of general information about the generated
                network (such as number of weights).
            no_weights: If set to True, no trainable parameters will be
                constructed, i.e., weights are assumed to be produced ad-hoc
                by a hypernetwork and passed to the forward function.
            network with a weight draw
                produced by the hypernetwork.
            num_heads: Number of output heads.
            init_weights (optional): This option is for convinience reasons.
                The option expects a list of parameter values that are used to
                initialize the network weights. As such, it provides a
                convinient way of initializing a
        """
        super(MLP, self).__init__(in_shape, num_classes, verbose)
        # first linear layer
        self._in_shape = in_shape
        self._all_shapes = [[layers[0]] + self._in_shape, [layers[0]]]
        # iterate through layers
        for i, o in zip(layers[:-1], layers[1:]):
            # add fully connected weight matrix
            self._all_shapes = self._all_shapes + [[o, i]]
            # add bias vector
            self._all_shapes = self._all_shapes + [[o]]
        # add last linear layer
        self._all_shapes += [[num_classes*num_heads, layers[-1]]] + \
                            [[num_classes*num_heads]]

        self._has_bias = True
        self._has_fc_out = True

        assert(init_weights is None or no_weights is False)
        self._no_weights = no_weights

        self._num_weights = Classifier.num_hyper_weights(self._all_shapes)
        if verbose:
            print('Creating a MLP with architecture %s and %d weights' \
                  % (str(self._all_shapes), self._num_weights))

        if no_weights:
            self._weights = None
            self._hyper_shapes = self._all_shapes
            self._is_properly_setup()
            return

        ### Define and initialize network weights.
        # Each odd entry of this list will contain a weight Tensor and each
        # even entry a bias vector.
        self._weights = nn.ParameterList()

        for i, dims in enumerate(self._all_shapes):
            self._weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))

        if init_weights is not None:
            assert(len(init_weights) == len(self._all_shapes))
            for i in range(len(init_weights)):
                assert(np.all(np.equal(list(init_weights[i].shape),
                                       list(self._weights[i].shape))))
                self._weights[i].data = init_weights[i]
        else:
            for i in range(0, len(self._weights), 2):
                init_params(self._weights[i], self._weights[i + 1])

        self._is_properly_setup()

    def forward(self, x, weights=None):
        """Predicts an activation vector y given x. If this network was
        constructed with no weights, then the passed weights are used.

        Args:
            x: The input to the network (e.g., an MNIST image).
            weights: A list of 6 weight tensors. And entries 0, 2, 4, ... are
                weight matrices for the fully connected weight matrices.
                The entries 1, 3, 5, ... are the respective bias vectors.
                would be:
                    fc1_W:   in_shape x arch[0]
                    fc1_b:   arch[0]
                    fc2_W:   arch[0] x arch[1]
                    fc2_b:   arch[1]
                    fc3_W:   arch[1] x num_classes
                    fc3_b:   num_classes
                    ...

        Returns:
            y
        """
        if self._no_weights and weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        if weights is not None:
            cur_weights = weights
        else:
            cur_weights = self._weights

        h = x.view(*([-1] + self._in_shape))
        #iterate through the d
        for i, j in zip(range(0, len(self._all_shapes), 2),
                        range(1, len(self._all_shapes), 2)):

            h = F.linear(h, cur_weights[i], bias=cur_weights[j])
            # non linear except from last layer
            if j !=  len(self._all_shapes) - 1:
                h = F.relu(h)
                
        return h

if __name__ == '__main__':
    pass
