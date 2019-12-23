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
- **title**          :utils/torch_utils.py
- **author**         :ch
- **contact**        :henningc@ethz.ch
- **created**        :09/11/2019
- **version**        :1.0
- **python_version** :3.6.8

A collection of helper functions that should capture common functionalities
needed when working with PyTorch.
"""
import math
import torch
from torch import nn

def init_params(weights, bias=None):
    """Initialize the weights and biases of a linear or (transpose) conv layer.

    Note, the implementation is based on the method "reset_parameters()",
    that defines the original PyTorch initialization for a linear or
    convolutional layer, resp. The implementations can be found here:

        https://git.io/fhnxV

        https://git.io/fhnx2

    Args:
        weights: The weight tensor to be initialized.
        bias (optional): The bias tensor to be initialized.
    """
    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)

def get_optimizer(params, lr, momentum=0, weight_decay=0, use_adam=False,
                  adam_beta1=0.9, use_rmsprop=False, use_adadelta=False,
                  use_adagrad=False):
    """Create an optimizer instance for the given set of parameters. Default
    optimizer is :class:`torch.optim.SGD`.

    Args:
        params: The parameters passed to the optimizer.
        lr: Learning rate.
        momentum (optional): Momentum (only applicable to
            :class:`torch.optim.SGD` and :class:`torch.optim.RMSprop`.
        weight_decay (optional): L2 penalty.
        use_adam: Use :class:`torch.optim.Adam` optimizer.
        adam_beta1: First parameter in the `betas` tuple that is passed to the
            optimizer :class:`torch.optim.Adam`:
            :code:`betas=(adam_beta1, 0.999)`.
        use_rmsprop: Use :class:`torch.optim.RMSprop` optimizer.
        use_adadelta: Use :class:`torch.optim.Adadelta` optimizer.
        use_adagrad: Use :class:`torch.optim.Adagrad` optimizer.

    Returns:
        Optimizer instance.
    """
    if use_adam:
        optimizer = torch.optim.Adam(params, lr=lr, betas=[adam_beta1, 0.999],
                                     weight_decay=weight_decay)
    elif use_rmsprop:
        optimizer = torch.optim.RMSprop(params, lr=lr,
                                        weight_decay=weight_decay,
                                        momentum=momentum)
    elif use_adadelta:
        optimizer = torch.optim.Adadelta(params, lr=lr,
                                         weight_decay=weight_decay)
    elif use_adagrad:
        optimizer = torch.optim.Adagrad(params, lr=lr,
                                        weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)

    return optimizer

if __name__ == '__main__':
    pass


