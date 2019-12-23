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
- **title**          :mnets/classifier_interface.py
- **author**         :ch
- **contact**        :henningc@ethz.ch
- **created**        :09/20/2019
- **version**        :1.0
- **python_version** :3.6.8

A general interface for main networks used in classification tasks. This
abstract base class also provides a collection of static helper functions that
are useful in classification problems.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from warnings import warn

from mnets.mnet_interface import MainNetInterface

class Classifier(nn.Module, MainNetInterface):
    """A general interface for classification networks.

    Attributes:
        num_classes: Number of output neurons.
    """
    def __init__(self, num_classes, verbose):
        """Initialize the network.

        Args:
            num_classes: The number of output neurons.
            verbose: Allow printing of general information about the generated
                network (such as number of weights).
        """
        # FIXME find a way using super to handle multiple inheritence.
        #super(Classifier, self).__init__()
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        assert(num_classes > 0)
        self._num_classes = num_classes

        self._verbose = verbose

    @property
    def num_classes(self):
        """Getter for read-only attribute num_classes."""
        return self._num_classes

    @staticmethod
    def logit_cross_entropy_loss(h, t, reduction='mean'):
        """Compute cross-entropy loss for given predictions and targets.
        Note, we assume that the argmax of the target vectors results in the
        correct label.

        Args:
            h: Unscaled outputs from the main network, i.e., activations of the
                last hidden layer (unscaled logits).
            t: Targets in form os soft labels or 1-hot encodings.
            reduction (str): The reduction method to be passed to
                :func:`torch.nn.functional.cross_entropy`.

        Returns:
            Cross-entropy loss computed on logits h and labels extracted
            from target vector t.
        """
        assert(t.shape[1] == h.shape[1])
        targets = t.argmax(dim=1, keepdim=False)
        return F.cross_entropy(h, targets, reduction=reduction)

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
                :code:`target_mapping = list(range(target_logits.shape[1]))`.
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
    def softmax_and_cross_entropy(h, t, reduction_sum=False):
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

        loss = -(t * torch.nn.functional.log_softmax(h, dim=1)).sum(dim=1)

        if reduction_sum:
            return loss.sum()
        else:
            return loss.mean()

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
    def num_hyper_weights(dims):
        """The number of weights that have to be predicted by a hypernetwork.

        .. deprecated:: 1.0
            Please use method
            :meth:`mnets.mnet_interface.MainNetInterface.shapes_to_num_weights`
            instead.

        Args:
            dims: For instance, the attribute :attr:`hyper_shapes`.

        Returns:
            (int)
        """
        warn('Please use class "mnets.mnet_interface.MainNetInterface.' +
             'shapes_to_num_weights" instead.', DeprecationWarning)

        return np.sum([np.prod(l) for l in dims])

if __name__ == '__main__':
    pass


