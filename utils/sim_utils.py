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
# @title          :utils/sim_utils.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :12/12/2019
# @version        :1.0
# @python_version :3.6.8
"""
General helper functions for simulations
----------------------------------------

The module :mod:`utils.sim_utils` comprises a bunch of functions that are in
general useful for writing simulations in this repository.
"""
import torch
import tensorboardX
from tensorboardX import SummaryWriter
import numpy as np
import random
import os
import shutil
import pickle
import logging
from time import time
from warnings import warn
import json

from cifar.sa_hyper_model import SAHyperNetwork
from mnets.mlp import MLP
from mnets.resnet import ResNet
from mnets.zenkenet import ZenkeNet
from mnist.chunked_hyper_model import ChunkedHyperNetworkHandler
from toy_example.hyper_model import HyperNetwork
from utils import logger_config
from utils import misc


def setup_environment(config, logger_name='hnet_sim_logger'):
    """Setup the general environment for training.

    This function should be called at the beginning of a simulation script
    (right after the command-line arguments have been parsed). The setup will
    incorporate:

        - creating the output folder
        - initializing logger
        - making computation deterministic (depending on config)
        - selecting the torch device
        - creating the Tensorboard writer

    Args:
        config (argparse.Namespace): Command-line arguments.

            .. note::
                The function expects command-line arguments available according
                to the function :func:`utils.cli_args.miscellaneous_args`.
        logger_name (str): Name of the logger to be created (time stamp will be
            appended to this name).

    Returns:
        (tuple): Tuple containing:

        - **device**: Torch device to be used.
        - **writer**: Tensorboard writer. Note, you still have to close the
          writer manually!
        - **logger**: Console (and file) logger.
    """
    ### Output folder.
    if os.path.exists(config.out_dir):
        # TODO allow continuing from an old checkpoint.
        response = input('The output folder %s already exists. ' % \
                         (config.out_dir) + \
                         'Do you want us to delete it? [y/n]')
        if response != 'y':
            raise Exception('Could not delete output folder!')
        shutil.rmtree(config.out_dir)

        os.makedirs(config.out_dir)
        print("Created output folder %s." % (config.out_dir))

    else:
        os.makedirs(config.out_dir)
        print("Created output folder %s." % (config.out_dir))

    # Save user configs to ensure reproducibility of this experiment.
    with open(os.path.join(config.out_dir, 'config.pickle'), 'wb') as f:
        pickle.dump(config, f)
    # A JSON file is easier to read for a human.
    with open(os.path.join(config.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f)

    ### Initialize logger.
    logger_name = '%s_%d' % (logger_name, int(time() * 1000))
    logger = logger_config.config_logger(logger_name,
        os.path.join(config.out_dir, 'logfile.txt'),
        logging.DEBUG, logging.INFO if config.loglevel_info else logging.DEBUG)
    # FIXME If we don't disable this, then the multiprocessing from the data
    # loader causes all messages to be logged twice. I could not find the cause
    # of this problem, but this simple switch fixes it.
    logger.propagate = False

    ### Deterministic computation.
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    # Ensure that runs are reproducible. Note, this slows down training!
    # https://pytorch.org/docs/stable/notes/randomness.html
    if config.deterministic_run:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if hasattr(config, 'num_workers') and config.num_workers > 1:
            logger.warning('Deterministic run desired but not possible with ' +
                           'more than 1 worker (see "num_workers").')

    ### Select torch device.
    assert(hasattr(config, 'no_cuda') or hasattr(config, 'use_cuda'))
    assert(not hasattr(config, 'no_cuda') or not hasattr(config, 'use_cuda'))

    if hasattr(config, 'no_cuda'):
        use_cuda = not config.no_cuda and torch.cuda.is_available()
    else:
        use_cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info('Using cuda: ' + str(use_cuda))

    ### Initialize summary writer.
    # Flushes every 120 secs by default.
    # DELETEME Ensure downwards compatibility.
    if not hasattr(tensorboardX, '__version__'):
        writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'summary'))
    else:
        writer = SummaryWriter(logdir=os.path.join(config.out_dir, 'summary'))

    return device, writer, logger

def get_mnet_model(config, net_type, in_shape, out_shape, device, cprefix=None,
                   no_weights=False):
    """Generate a main network instance.

    A helper to generate a main network according to the given the user
    configurations.

    .. note::
        Generation of networks with context-modulation is not yet supported,
        since there is no global argument set in :mod:`utils.cli_args` yet.

    Args:
        config (argparse.Namespace): Command-line arguments.

            .. note::
                The function expects command-line arguments available according
                to the function :func:`utils.cli_args.main_net_args`.
        net_type (str): The type of network. The following options are
            available:
            
            - ``mlp``: :class:`mnets.mlp.MLP`
            - ``resnet``: :class:`mnets.resnet.ResNet`
            - ``zenke``: :class:`mnets.zenkenet.ZenkeNet`
            - ``bio_conv_net``: :class:`mnets.bio_conv_net.BioConvNet`
        in_shape (list): Shape of network inputs. Can be ``None`` if not
            required by network type.

            For instance: For an MLP network :class:`mnets.mlp.MLP` with 100
            input neurons it should be :code:`in_shape=[100]`.
        out_shape (list): Shape of network outputs. See ``in_shape`` for more
            details.
        device: PyTorch device.
        cprefix (str, optional): A prefix of the config names. It might be, that
            the config names used in this method are prefixed, since several
            main networks should be generated (e.g., :code:`cprefix='gen_'` or
            ``'dis_'`` when training a GAN).

            Also see docstring of parameter ``prefix`` in function
            :func:`utils.cli_args.main_net_args`.
        no_weights (bool): Whether the main network should be generated without
            weights.

    Returns:
        The created main network model.
    """
    assert(net_type in ['mlp', 'resnet', 'zenke', 'bio_conv_net'])

    if cprefix is None:
        cprefix = ''

    def gc(name):
        """Get config value with that name."""
        return getattr(config, '%s%s' % (cprefix, name))
    def hc(name):
        """Check whether config exists."""
        return hasattr(config, '%s%s' % (cprefix, name))

    mnet = None

    if hc('net_act'):
        net_act = gc('net_act')
        net_act = misc.str_to_act(net_act)
    else:
        net_act = None

    def get_val(name):
        ret = None
        if hc(name):
            ret = gc(name)
        return ret

    no_bias = get_val('no_bias')
    dropout_rate = get_val('dropout_rate')
    specnorm = get_val('specnorm')
    batchnorm = get_val('batchnorm')
    no_batchnorm = get_val('no_batchnorm')
    bn_no_running_stats = get_val('bn_no_running_stats')
    bn_distill_stats = get_val('bn_distill_stats')
    #bn_no_stats_checkpointing = get_val('bn_no_stats_checkpointing')

    use_bn = None
    if batchnorm is not None:
        use_bn = batchnorm
    elif no_batchnorm is not None:
        use_bn = not no_batchnorm

    # FIXME if an argument wasn't specified, then we use the default value that
    # is currently (at time of implementation) in the constructor.
    assign = lambda x, y : y if x is None else x

    if net_type == 'mlp':
        assert(hc('mlp_arch'))
        assert(len(in_shape) == 1 and len(out_shape) == 1)

        mnet = MLP(n_in=in_shape[0], n_out=out_shape[0],
            hidden_layers=misc.str_to_ints(gc('mlp_arch')),
            activation_fn=assign(net_act, torch.nn.ReLU()),
            use_bias=not assign(no_bias, False),
            no_weights=no_weights,
            #init_weights=None,
            dropout_rate=assign(dropout_rate, -1),
            use_spectral_norm=assign(specnorm, False),
            use_batch_norm=assign(use_bn, False),
            bn_track_stats=assign(not bn_no_running_stats, True),
            distill_bn_stats=assign(bn_distill_stats, False),
            #use_context_mod=False,
            #context_mod_inputs=False,
            #no_last_layer_context_mod=False,
            #context_mod_no_weights=False,
            #context_mod_post_activation=False,
            #context_mod_gain_offset=False,
            #out_fn=None,
            verbose=True).to(device)

    elif net_type == 'resnet':
        assert(len(out_shape) == 1)

        mnet = ResNet(in_shape=in_shape, num_classes=out_shape[0],
            verbose=True, #n=5,
            no_weights=no_weights,
            #init_weights=None,
            use_batch_norm=assign(use_bn, True),
            bn_track_stats=assign(not bn_no_running_stats, True),
            distill_bn_stats=assign(bn_distill_stats, False),
            #use_context_mod=False,
            #context_mod_inputs=False,
            #no_last_layer_context_mod=False,
            #context_mod_no_weights=False,
            #context_mod_post_activation=False,
            #context_mod_gain_offset=False,
            #context_mod_apply_pixel_wise=False
        ).to(device)

    elif net_type == 'zenke':
        assert(len(out_shape) == 1)

        mnet = ZenkeNet(in_shape=in_shape, num_classes=out_shape[0],
            verbose=True, #arch='cifar',
            no_weights=no_weights,
            #init_weights=None,
            dropout_rate=assign(dropout_rate, 0.25)).to(device)
    else:
        assert(net_type == 'bio_conv_net')
        assert(len(out_shape) == 1)

        raise NotImplementedError('Implementation not publicly available!')

    return mnet

def get_hnet_model(config, num_tasks, device, mnet_shapes, cprefix=None):
    """Generate a hypernetwork instance.

    A helper to generate the hypernetwork according to the given the user
    configurations.

    Args:
        config (argparse.Namespace): Command-line arguments.

            .. note::
                The function expects command-line arguments available according
                to the function :func:`utils.cli_args.hypernet_args`.
        num_tasks (int): The number of task embeddings the hypernetwork should
            have.
        device: PyTorch device.
        mnet_shapes: Dimensions of the weight tensors of the main network.
            See main net argument
            :attr:`mnets.mnet_interface.MainNetInterface.param_shapes`.
        cprefix (str, optional): A prefix of the config names. It might be, that
            the config names used in this method are prefixed, since several
            hypernetworks should be generated (e.g., :code:`cprefix='gen_'` or
            ``'dis_'`` when training a GAN).

            Also see docstring of parameter ``prefix`` in function
            :func:`utils.cli_args.hypernet_args`.

    Returns:
        The created hypernet model.
    """
    if cprefix is None:
        cprefix = ''

    def gc(name):
        """Get config value with that name."""
        return getattr(config, '%s%s' % (cprefix, name))

    hyper_chunks = misc.str_to_ints(gc('hyper_chunks'))
    assert(len(hyper_chunks) in [1,2,3])
    if len(hyper_chunks) == 1:
        hyper_chunks = hyper_chunks[0]

    hnet_arch = misc.str_to_ints(gc('hnet_arch'))
    sa_hnet_filters = misc.str_to_ints(gc('sa_hnet_filters'))
    sa_hnet_kernels = misc.str_to_ints(gc('sa_hnet_kernels'))
    sa_hnet_attention_layers = misc.str_to_ints(gc('sa_hnet_attention_layers'))

    hnet_act = misc.str_to_act(gc('hnet_act'))

    if isinstance(hyper_chunks, list): # Chunked self-attention hypernet
        if len(sa_hnet_kernels) == 1:
            sa_hnet_kernels = sa_hnet_kernels[0]
        # Note, that the user can specify the kernel size for each dimension and
        # layer separately.
        elif len(sa_hnet_kernels) > 2 and \
            len(sa_hnet_kernels) == gc('sa_hnet_num_layers') * 2:
            tmp = sa_hnet_kernels
            sa_hnet_kernels = []
            for i in range(0, len(tmp), 2):
                sa_hnet_kernels.append([tmp[i], tmp[i+1]])

        if gc('hnet_dropout_rate') != -1:
            warn('SA-Hypernet doesn\'t use dropout. Dropout rate will be ' +
                 'ignored.')
        if gc('hnet_act') != 'relu':
            warn('SA-Hypernet doesn\'t support the other non-linearities ' +
                 'than ReLUs yet. Option "%shnet_act" (%s) will be ignored.'
                 % (cprefix, gc('hnet_act')))

        hnet = SAHyperNetwork(mnet_shapes, num_tasks,
            out_size=hyper_chunks,
            num_layers=gc('sa_hnet_num_layers'),
            num_filters=sa_hnet_filters,
            kernel_size=sa_hnet_kernels,
            sa_units=sa_hnet_attention_layers,
            # Note, we don't use an additional hypernet for the remaining
            # weights!
            #rem_layers=hnet_arch,
            te_dim=gc('temb_size'),
            ce_dim=gc('emb_size'),
            no_theta=False,
            # Batchnorm and spectral norma are not yet implemented.
            #use_batch_norm=gc('hnet_batchnorm'),
            #use_spectral_norm=gc('hnet_specnorm'),
            # Droput would only be used for the additional network, which we
            # don't use.
            #dropout_rate=gc('hnet_dropout_rate'),
            discard_remainder=True,
            noise_dim=gc('hnet_noise_dim'),
            temb_std=gc('temb_std')).to(device)

    elif hyper_chunks != -1: # Chunked fully-connected hypernet
        hnet = ChunkedHyperNetworkHandler(mnet_shapes, num_tasks,
            chunk_dim=hyper_chunks, layers=hnet_arch,
            activation_fn=hnet_act, te_dim=gc('temb_size'),
            ce_dim=gc('emb_size'), dropout_rate=gc('hnet_dropout_rate'),
            noise_dim=gc('hnet_noise_dim'),
            temb_std=gc('temb_std')).to(device)

    else: # Fully-connected hypernet.
        hnet = HyperNetwork(mnet_shapes, num_tasks, layers=hnet_arch,
            te_dim=gc('temb_size'), activation_fn=hnet_act,
            dropout_rate=gc('hnet_dropout_rate'),
            noise_dim=gc('hnet_noise_dim'),
            temb_std=gc('temb_std')).to(device)

    return hnet

if __name__ == '__main__':
    pass


