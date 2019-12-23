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
#
# @title          :utils/init_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :12/20/2019
# @version        :1.0
# @python_version :3.6.9
"""
Helper functions for weight initialization
------------------------------------------

The module :mod:`utils.init_utils` contains helper functions that might be
useful for initialization of weights. The functions are somewhat complementary
to what is already provided in the PyTorch module :mod:`torch.nn.init`.
"""
import math
import numpy as np
import torch

def xavier_fan_in_(tensor):
    """Initialize the given weight tensor with Xavier fan-in init.

    Unfortunately, :func:`torch.nn.init.xavier_uniform_` doesn't give
    us the choice to use fan-in init (always uses the harmonic mean).
    Therefore, we provide our own implementation.

    Args:
        tensor (torch.Tensor): Weight tensor that will be modified
            (initialized) in-place.
    """
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = 1. / math.sqrt(fan_in)
    # Note, std(Unif(-a, a)) = a / sqrt(3)
    a = math.sqrt(3.0) * std

    torch.nn.init._no_grad_uniform_(tensor, -a, a)

def calc_fan_in_and_out(shapes):
    """Calculate fan-in and fan-out.

    Note:
        This function expects the shapes of an at least 2D tensor.

    Args:
        shapes (list): List of integers.

    Returns:
        (tuple) Tuple containing:

        - **fan_in**
        - **fan_out**
    """
    assert len(shapes) > 1
    
    fan_in = shapes[1]
    fan_out = shapes[0]

    if len(shapes) > 2:
        receptive_field_size = int(np.prod(shapes[2:]))
    else:
        receptive_field_size = 1

    fan_in *= receptive_field_size
    fan_out *= receptive_field_size

    return fan_in, fan_out

if __name__ == '__main__':
    pass


