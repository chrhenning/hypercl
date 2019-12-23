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
@title           :utils/misc.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :04/11/2019
@version         :1.0
@python_version  :3.6.7

A collection of helper functions.
"""

import matplotlib
import matplotlib.pyplot as plt
import math
from torch import nn
import torch
from warnings import warn

def init_params(weights, bias=None):
    """Initialize the weights and biases of a linear or (transpose) conv layer.

    Note, the implementation is based on the method "reset_parameters()",
    that defines the original PyTorch initialization for a linear or
    convolutional layer, resp. The implementations can be found here:

        https://git.io/fhnxV

        https://git.io/fhnx2

    .. deprecated:: 1.0
        Please use function :func:`utils.torch_utils.init_params` instead.

    Args:
        weights: The weight tensor to be initialized.
        bias (optional): The bias tensor to be initialized.
    """
    warn('Function is deprecated. Use "utils.torch_utils.init_params" instead.',
         DeprecationWarning)

    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)

def str_to_ints(str_arg):
    """Helper function to convert a list of comma separated strings into
    integers.

    Args:
        str_arg: String containing list of comma-separated ints. For convenience
            reasons, we allow the user to also pass single integers that a put
            into a list of length 1 by this function.

    Returns:
        List of integers.
    """
    if isinstance(str_arg, int):
        return [str_arg]

    if len(str_arg) > 0:
        return [int(s) for s in str_arg.split(',')]
    else:
        return []

def list_to_str(list_arg, delim=' '):
    """Convert a list of numbers into a string.

    Args:
        list_arg: List of numbers.
        delim (optional): Delimiter between numbers.

    Returns:
        List converted to string.
    """
    ret = ''
    for i, e in enumerate(list_arg):
        if i > 0:
            ret += delim
        ret += str(e)
    return ret

def str_to_act(act_str):
    """Convert the name of an activation function into the actual PyTorch
    activation function.

    Args:
        act_str: Name of activation function (as defined by command-line
            arguments).

    Returns:
        Torch activation function instance or ``None``, if ``linear`` is given.
    """
    if act_str == 'linear':
        act = None
    elif act_str == 'sigmoid':
        act = torch.nn.Sigmoid()
    elif act_str == 'relu':
        act = torch.nn.ReLU()
    elif act_str == 'elu':
        act = torch.nn.ELU()
    else:
        raise Exception('Activation function %s unknown.' % act_str)
    return act

def configure_matplotlib_params(fig_size = [6.4, 4.8], two_axes=True,
                                font_size=8):
    """Helper function to configure default matplotlib parameters.

    Args:
        fig_size: Figure size (width, height) in inches.
    """
    params = {
        'axes.labelsize': font_size,
        'font.size': font_size,
        'font.sans-serif': ['Arial'],
        'text.usetex': True,
        'text.latex.preamble': [r'\usepackage[scaled]{helvet}',
                                r'\usepackage{sfmath}'],
        'font.family': 'sans-serif',
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'axes.titlesize': font_size,
        'axes.spines.right' : not two_axes,
        'axes.spines.top' : not two_axes,
        'figure.figsize': fig_size,
        'legend.handlelength': 0.5
    }

    matplotlib.rcParams.update(params)

def get_colorbrewer2_colors(family = 'Set2'):
    """Helper function that returns a list of color combinations
    extracted from colorbrewer2.org.

    Args:
        type: the color family from colorbrewer2.org to use.
    """
    if family == 'Set2':
        return [
            '#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3',
            '#ff7f00',
            '#ffff33',
            '#a65628',
            '#b3de69'
        ]
    if family == 'Set3':
        return [
            '#8dd3c7',
            '#ffffb3',
            '#bebada',
            '#fb8072',
            '#80b1d3',
            '#fdb462',
            ''
        ]
    elif family == 'Dark2':
        return [
            '#1b9e77',
            '#d95f02',
            '#7570b3',
            '#e7298a',
            '#66a61e',
            '#e6ab02',
            '#a6761d'
        ]
    elif family == 'Pastel':
        return [
            '#fbb4ae',
            '#b3cde3',
            '#ccebc5',
            '#decbe4',
            '#fed9a6',
            '#ffffcc',
            '#e5d8bd'
        ]

def repair_canvas_and_show_fig(fig, close=True):
    """If writing a figure to tensorboard via "add_figure" it might change the
    canvas, such that our backend doesn't allow to show the figure anymore.
    This method will generate a new canvas and replace the old one of the
    given figure.

    Args:
        fig: The figure to be shown.
        close: Whether the figure should be closed after it has been shown.
    """
    tmp_fig = plt.figure()
    tmp_manager = tmp_fig.canvas.manager
    tmp_manager.canvas.figure = fig
    fig.set_canvas(tmp_manager.canvas)
    plt.close(tmp_fig.number)
    plt.figure(fig.number)
    plt.show()
    if close:
        plt.close(fig.number)

if __name__ == '__main__':
    pass