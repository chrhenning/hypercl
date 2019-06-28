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
@title           :main_model.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :10/25/2018
@version         :1.0
@python_version  :3.6.6

This module contains a general classifier template and a LeNet-like network
to classify either MNIST or CIFAR-10 images. The network is implemented in a
way that it might not have trainable parameters. Instead, the network weights
would have to be passed to the forward function. This makes the usage of a
hypernetwork (a network that generates the weights of another network)
particularly easy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from warnings import warn

from utils.module_wrappers import MainNetInterface
from utils.misc import init_params

class Classifier(nn.Module, MainNetInterface):
    """A general interface for classification networks.

    Attributes (additional to base class):
        num_classes: Number of output neurons.
    """
    def __init__(self, in_shape, num_classes, verbose):
        """Initialize the network.

        Args:
            in_shape: The shape of an input sample. Note, we assume the
                Tensorflow format, where the last entry denotes the number of
                channels.
            num_classes: The number of output neurons.
            verbose: Allow printing of general information about the generated
                network (such as number of weights).
        """
        # FIXME find a way using super to handle multiple inheritence.
        #super(Classifier, self).__init__()
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        self._in_shape = in_shape

        assert(num_classes > 0)
        self._num_classes = num_classes

        self._verbose = verbose

    @property
    def num_classes(self):
        """Getter for read-only attribute num_classes."""
        return self._num_classes

    @staticmethod
    def cross_entropy_loss(h, t):
        """Deprecated, use "logit_cross_entropy_loss" instead."""
        warn('Use "logit_cross_entropy_loss" instead.', DeprecationWarning)
        Classifier.logit_cross_entropy_loss(h, t)

    @staticmethod
    def logit_cross_entropy_loss(h, t):
        """Compute cross-entropy loss for given predictions and targets.
        Note, we assume that the argmax of the target vectors results in the
        correct label.

        Args:
            h: Unscaled outputs from the main network, i.e., activations of the
                last hidden layer (unscaled logits).
            t: Targets in form os soft labels or 1-hot encodings.

        Returns:
            Cross-entropy loss computed on logits h and labels extracted
            from target vector t.
        """
        assert(t.shape[1] == h.shape[1])
        targets = t.argmax(dim=1, keepdim=False)
        return F.cross_entropy(h, targets)

    @staticmethod
    def knowledge_distillation_loss(logits, target_logits, target_mapping=None,
                                    device=None, T=2.):
        """Compute the knowledge distillation loss as proposed by
            Hinton et al., "Distilling the Knowledge in a Neural Network",
            NIPS Deep Learning and Representation Learning Workshop, 2015.
            http://arxiv.org/abs/1503.02531

        Args:
            logits: Unscaled outputs from the main network, i.e., activations of
                the last hidden layer (unscaled logits).
            target_logits: Target logits, i.e., activations of the last hidden
                layer (unscaled logits) from the target model.
                Note, we won't detach "target_logits" from the graph. Make sure,
                that you do this before calling this method.
            target_mapping: In continual learning, it might be that the output
                layer size of a model is growing. Thus, it could be that the
                model providing the "target_logits" has a smaller output size
                than the current model providing the "logits". Therefore, one
                has to provide a mapping, which is a list of indices for
                "logits" that state which activations in "logits" have a
                corresponding target in "target_logits".
                For instance, if the output layer size just increased by 1
                through appending a new output neuron to the current model, the
                mapping would simply be:
                    target_mapping = list(range(target_logits.shape[1]))
            device: Current PyTorch device. Only needs to be specified if
                "target_mapping" is given.
            T: Softmax temperature.

        Returns:
            Knowledge Distillation (KD) loss.
        """
        assert(target_mapping is None or device is not None)
        targets = F.softmax(target_logits / T, dim=1)
        n_classes = logits.shape[1]
        n_targets = targets.shape[1]

        if target_mapping is None:
            if n_classes != n_targets:
                raise ValueError('If sizes of "logits" and "target_logits" ' +
                                 'differ, "target_mapping" must be specified.')
        else:
            new_targets = torch.zeros_like(logits).to(device)
            new_targets[:, target_mapping] = targets
            targets = new_targets

        return -(targets * F.log_softmax(logits / T,dim=1)).sum(dim=1).mean()*\
               T**2

    @staticmethod
    def softmax_and_cross_entropy(h, t):
        """Compute the cross entropy from logits, allowing smoothed labels
        (i.e., this function does not require 1-hot targets).

        Args:
            h: Unscaled outputs from the main network, i.e., activations of the
                last hidden layer (unscaled logits).
            t: Targets in form os soft labels or 1-hot encodings.

        Returns:
            Cross-entropy loss computed on logits h and given targets t.
        """
        assert(t.shape[1] == h.shape[1])
        return -(t * torch.nn.functional.log_softmax(h, dim=1)).sum(dim=1). \
            mean()

    @staticmethod
    def accuracy(y, t):
        """Computing the accuracy between predictions y and targets t. We
        assume that the argmax of t results in labels as described in the
        docstring of method "cross_entropy_loss".

        Args:
            y: Outputs from the main network.
            t: Targets in form of soft labels or 1-hot encodings.

        Returns:
            Relative prediction accuracy on the given batch.
        """
        assert(t.shape[1] == y.shape[1])
        predictions = y.argmax(dim=1, keepdim=False)
        targets = t.argmax(dim=1, keepdim=False)

        return (predictions == targets).float().mean()

    @staticmethod
    def _init_params(weights, bias=None):
        """Initialize the weights and biases of a linear or conv2d layer.

        Note, the implementation is based on the method "reset_parameters()",
        that defines the original PyTorch initialization for a linear or
        convolutional layer, resp. The implementations can be found here:
            https://git.io/fhnxV
            https://git.io/fhnx2

        Args:
            weights: The weight tensor to be initialized.
            bias (optional): The bias tensor to be initialized.
        """
        warn('Use "utils.misc.init_params" instead.', DeprecationWarning)

        nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

    @staticmethod
    def num_hyper_weights(dims):
        """The number of weights that have to be predicted by a hypernetwork.

        Args:
            dims: For instance, the attribute "hyper_shapes".

        Returns:
            Integer.
        """
        return np.sum([np.prod(l) for l in dims])

class ZenkeNet(Classifier):
    """The network consists of four convolutional layers followed by two fully-
    connected layers. See implementation for details.

    ZenkeNet is a network introduced in 
        "Continual Learning Through Synaptic Intelligence", Zenke et
        al., 2017.
    See Appendix for details.
    We use the same network for comparison to the results reported in the paper.

    Attributes (additional to base class):
    """
    _architectures = { 
        'cifar': [[32,3,3,3],[32],[32,32,3,3],[32],[64,32,3,3],[64],
                  [64,64,3,3],[64],[512, 2304],[512],[10,512],[10]]
        }

    def __init__(self, in_shape=[32, 32, 3],
                 num_classes=10, verbose=True, arch='cifar',
                 no_weights=False, init_weights=None, 
                 num_heads=1, use_dropout=True,
                 dropout_prob=0.25):
        """Initialize the network.

        Args:
            in_shape: The shape of an input sample. Note, we assume the
                Tensorflow format, where the last entry denotes the number of
                channels.
            num_classes: The number of output neurons. The chosen architecture
                ("arch") will be adopted accordingly.
            verbose: Allow printing of general information about the generated
                network (such as number of weights).
            arch: The architecture to be employed. There are three options
                available:
                    "cifar": A Convolutional network used by Zenke in his
                             for his split CIFAR-10 / 100 experiment.
            no_weights: If set to True, no trainable parameters will be
                constructed, i.e., weights are assumed to be produced ad-hoc
                by a hypernetwork and passed to the forward function.
            init_weights (optional): This option is for convinience reasons.
                The option expects a list of parameter values that are used to
                initialize the network weights. As such, it provides a
                convinient way of initializing a network with a weight draw
                produced by the hypernetwork.
            num_heads: Number of output heads.
            use_dropout: Whether to use dropout in this network. If activated,
                dropout will be applied after the second convolutional layer
                (before pooling) and after the first fully-connected layer
                (after the activation function).
            dropout_prob (default: 0.25): The dropout probability, if dropout is
                used.
        """
        super(ZenkeNet, self).__init__(in_shape, num_classes*num_heads, verbose)

        assert(self._in_shape[0] == 32 and self._in_shape[1] == 32)

        assert(arch in ZenkeNet._architectures.keys())
        self._all_shapes = ZenkeNet._architectures[arch]

        self._all_shapes[-2][0] = num_classes*num_heads
        self._all_shapes[-1][0] = num_classes*num_heads

        assert(init_weights is None or no_weights is False)
        self._no_weights = no_weights

        self._use_dropout = use_dropout

        self._has_bias = True
        self._has_fc_out = True

        self._num_weights = Classifier.num_hyper_weights(self._all_shapes)
        if verbose:
            print('Creating a ZenkeNet with %d weights' \
                  % (self._num_weights)
                  + (', that uses dropout.' if use_dropout else '.'))

        if use_dropout:
            # Empirically, it seemed to work better, if there is no dropout
            # after the first convolutional layer.
            #self._drop_conv1 = nn.Dropout2d(p=dropout_prob)
            self._drop_conv2 = nn.Dropout2d(p=dropout_prob)
            self._drop_fc1 = nn.Dropout(p=dropout_prob*2)

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
                init_params(self._weights[i], self._weights[i+1])

        self._is_properly_setup()

    def forward(self, x, weights=None):
        """Predicts an activation vector y given x. If this network was
        constructed with no weights, then the passed weights are used.

        Args:
            x: The input to the network (e.g., an MNIST image).
            weights: A list of weight tensors, where odd entries are supposed to
                correspond to bias vectors.

        Returns:
            y: The output of the network.
        """
        if self._no_weights and weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        if not self._no_weights:
            if weights is not None:
                raise Exception('The network was created with trainable ' +
                                'weights, thus "weights" may not be passed ' +
                                'to the "forward" function.')
            weights = self._weights
        else:
            shapes = self.weight_shapes()
            assert(len(weights) == len(shapes))
            for i, s in enumerate(shapes):
                assert(np.all(np.equal(s, list(weights[i].shape))))

        # Note, implementation aims to follow:
        #     https://git.io/fj8xP

        # first block
        x = x.view(*([-1] + self._in_shape))
        x = x.permute(0, 3, 1, 2)
        h = F.conv2d(x, weights[0], bias=weights[1], padding=1) # 'SAME'
        h = F.relu(h)
        h = F.conv2d(h, weights[2], bias=weights[3], padding=0) # 'VALID'
        h = F.max_pool2d(F.relu(h), 2)
        if self._use_dropout:
            h = self._drop_conv2(h)
        
        # second block
        h = F.conv2d(h, weights[4], bias=weights[5], padding=1) # 'SAME'
        h = F.relu(h)
        h = F.conv2d(h, weights[6], bias=weights[7], padding=0) # 'VALID'
        h = F.max_pool2d(F.relu(h), 2) 
        if self._use_dropout:
            h = self._drop_conv2(h)

        # last fully connected layers
        h = h.view(-1, weights[8].size()[1])
        h = F.relu(F.linear(h, weights[8], bias=weights[9]))
        if self._use_dropout:
            h = self._drop_fc1(h)
        h = F.linear(h, weights[10], bias=weights[11])

        return h

if __name__ == '__main__':
    pass
