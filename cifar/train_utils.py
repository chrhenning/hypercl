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
# @title           :classifier/train_utils.py
# @author          :ch, jvo
# @contact         :henningc@ethz.ch
# @created         :02/26/2019
# @version         :1.0
# @python_version  :3.6.6
"""
Helper functions for training CIFAR experiments via deterministic CL
--------------------------------------------------------------------

The module :mod:`cifar.train_utils` contains a collection of helper methods for
the module :mod:`cifar.train`. The reason why these nethods are outsourced to
this module is simply to improve readibility of the module :mod:`cifar.train`.
The methods collected here are typically not required to understand the
underlying logic of the training process.
"""
import os
import torch

from data.special.split_cifar import get_split_cifar_handlers
from mnist.chunked_hyper_model import ChunkedHyperNetworkHandler
from mnets.mnet_interface import MainNetInterface
from utils import misc
from utils import sim_utils as sutils

def load_datasets(config, shared, logger, data_dir='../datasets'):
    """Create a data handler per task.

    Note:
        Datasets are generated with targets being 1-hot encoded.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Object for sharing data between functions.
            Contains the type of experiment.
        logger: Logger object.
        data_dir (str): From where to load (or to where to download) the
            datasets?

    Returns:
        (list) A list of data handlers (i.e., objects of class
        :class:`data.dataset.Dataset`.
    """
    augment_data = not config.disable_data_augmentation
    #if shared.experiment == 'zenke':
    #    augment_data = False
    #    # To be comparable to previous results. Note, Zenke et al. didn't
    #    # utilize any data augmentation as far as I know.
    #    logger.warning('Data augmentation disabled for Zenkenet.')
    if augment_data:
        logger.info('Data augmentation will be used.')

    assert(config.num_tasks <= 11)
    logger.info('Loading CIFAR datasets ...')
    dhandlers = get_split_cifar_handlers(data_dir, use_one_hot=True,
        use_data_augmentation=augment_data, num_tasks=config.num_tasks)
    assert(len(dhandlers) == config.num_tasks)

    logger.info('Loaded %d CIFAR task(s) into memory.' % config.num_tasks)

    return dhandlers

def get_main_model(config, shared, logger, device, no_weights=False):
    """Helper function to generate the main network.

    This function uses :func:`utils.sim_utils.get_mnet_model` to generate the
    main network.

    The function also takes care of weight initialization, if configured.

    Args:
        (....): See docstring of function :func:`load_datasets`.
        device: The PyTorch device.
        no_weights (bool): If ``True``, the main network is generated without
            internal weights.

    Returns:
        The main network.
    """
    if shared.experiment == 'zenke':
        net_type = 'zenke'
        logger.info('Building a ZenkeNet ...')

    else:
        net_type = 'resnet'
        logger.info('Building a ResNet ...')

    num_outputs = 10

    if config.cl_scenario == 1 or config.cl_scenario == 3:
        num_outputs *= config.num_tasks

    logger.info('The network will have %d output neurons.' % num_outputs)

    in_shape = [32, 32, 3]
    out_shape = [num_outputs]

    # TODO Allow main net only training.
    mnet =  sutils.get_mnet_model(config, net_type, in_shape, out_shape, device,
                                  no_weights=no_weights)

    init_network_weights(mnet.weights, config, logger, net=mnet)

    return mnet

def get_hnet_model(config, mnet, logger, device):
    """Generate the hypernetwork.

    This function uses :func:`utils.sim_utils.get_hnet_model` to generate the
    hypernetwork.

    The function also takes care of weight initialization, if configured.

    Args:
        (....): See docstring of function :func:`get_main_model`.
        mnet: The main network.

    Returns:
        The hypernetwork or ``None`` if no hypernet is needed.
    
    """
    logger.info('Creating hypernetwork ...')
    hnet = sutils.get_hnet_model(config, config.num_tasks, device,
                                 mnet.param_shapes)
    # FIXME There should be a nicer way of initializing hypernets in the
    # future.
    chunk_embs = None
    if hasattr(hnet, 'chunk_embeddings'):
        chunk_embs = hnet.chunk_embeddings
    init_network_weights(hnet.parameters(), config, logger,
        chunk_embs=chunk_embs, task_embs=hnet.get_task_embs(), net=hnet)
    if config.hnet_init_shift:
        hnet_init_shift(hnet, mnet, config, logger, device)

    # TODO Incorporate hyperchunk init.
    #if isinstance(hnet, ChunkedHyperNetworkHandler):
    #    hnet.apply_chunked_hyperfan_init(temb_var=config.std_normal_temb**2)

    return hnet

def init_network_weights(all_params, config, logger, chunk_embs=None,
                         task_embs=None, net=None):
    """Initialize a given set of weight tensors according to the user
    configuration.

    Warning:
        This method is agnostic to where the weights stem from and is
        therefore slightly dangerous. Use with care.

    Note:
        The method only exists as at the time of implementation the package
        :mod:`hnets` wasn't available yet. In the future, initialization should
        be part of the network implementation (e.g., via method
        :meth:`mnets.mnet_interface.MainNetInterface.custom_init`).

    Note:
        If the given network implements interface
        :class:`mnets.mnet_interface.MainNetInterface`, then the corresponding
        method :meth:`mnets.mnet_interface.MainNetInterface.custom_init` is
        used.

    Note:
        Papers like the following show that hypernets should get a special
        init. This function does not take this into consideration.

            https://openreview.net/forum?id=H1lma24tPB

    Args:
        all_params: A list of weight tensors to be initialized.
        config: Command-line arguments.
        logger: Logger.
        chunk_embs (optional): A list of chunk embeddings.
        task_embs (optional): A list of task embeddings.
        net (optional): The network from which the parameters stem come from.
            Can be used to implement network specific initializations (e.g.,
            batch-norm weights).
    """
    if config.custom_network_init:
        if net is not None and isinstance(net, MainNetInterface):
            logger.info('Applying custom initialization to network ...')
            net.custom_init(normal_init=config.normal_init,
                            normal_std=config.std_normal_init, zero_bias=True)

        else:
            logger.warning('Custom weight initialization is applied to all ' +
                           'network parameters. Note, the current ' +
                           'implementation might be agnostic to special ' +
                           'network parameters.')
            for W in all_params:
                # FIXME not all 1D vectors are bias vectors.
                # Examples of parameters that are 1D and not bias vectors:
                # * batchnorm weights
                # * embedding vectors
                if W.ndimension() == 1: # Bias vector.
                    torch.nn.init.constant_(W, 0)
                elif config.normal_init:
                    torch.nn.init.normal_(W, mean=0, std=config.std_normal_init)
                else:
                    torch.nn.init.xavier_uniform_(W)


    # Note, the embedding vectors inside "all_params" have been considered
    # as bias vectors and thus initialized to zero.
    if chunk_embs is not None:
        for emb in chunk_embs:
            torch.nn.init.normal_(emb, mean=0, std=config.std_normal_emb)

    if task_embs is not None:
        for temb in task_embs:
            torch.nn.init.normal_(temb, mean=0., std=config.std_normal_temb)

def hnet_init_shift(hnet, mnet, config, logger, device):
    """Init the hypernet ``hnet`` such that the weights of the main network 
    ``mnet`` are initialised as if there would be no hypernetwork i.e. the first
    hypernetwork output is a standard init (for now normal or Xavier
    are implemented).

    Note:
        This function is only meant for exploratory purposes. It does not
        provide a proper weight initialization as for instance

            https://openreview.net/forum?id=H1lma24tPB

        Though, it is independent of the hypernet type/architecture.

    Warning:
        Not all hypernets support this quick-fix.

    Args:
        hnet: The model of the hyper network.
        mnet: The main model.
        config: The command line arguments.
        device: Torch device (cpu or gpu).
    """
    logger.warning('Config "hnet_init_shift" is just a temporary test and ' +
                   'should be used with care.')

    # Get the current output, this should be normal or xavier or ...
    hnet_outputs = hnet.forward(0)
    orig_output = [o.detach().clone() for o in hnet_outputs]
    mnet_init = [torch.zeros_like(o) for o in hnet_outputs]

    tmp = config.custom_network_init
    config.custom_network_init = True
    init_network_weights(mnet_init, config, logger, net=mnet)
    config.custom_network_init = tmp

    # The shift of the hypernetwork outputs will be computed by subtracting the
    # current output and adding the new desired output.
    o_shift = []
    for i, o in enumerate(orig_output):
        o_shift.append(-o + mnet_init[i])

    # set the shifts
    assert(hasattr(hnet, '_shifts')) # Only temporarily added to some hnets.
    hnet._shifts = o_shift

def setup_summary_dict(config, shared, mnet, hnet=None):
    """Setup the summary dictionary that is written to the performance
    summary file (in the result folder).

    This method adds the keyword ``summary`` to ``shared``.

    Args:
        config (argparse.Namespace): Command-line arguments.
        shared (argparse.Namespace): Miscellaneous data shared among training
            functions (summary dict will be added to this object).
        mnet: Main network.
        hnet (optional): Hypernetwork.
    """
    assert(hasattr(shared, 'experiment'))

    summary = dict()

    if hnet is None:
        num = mnet.num_params
        hnum = -1
        ratio = -1
    else:
        num = hnet.num_outputs
        hnum = hnet.num_weights
        ratio = hnum / num

    # FIXME keywords should be cross-checked with those specified in the
    # corresponding `_SUMMARY_KEYWORDS` of the hyperparam search.

    summary['acc_final'] = [-1] * config.num_tasks
    summary['acc_during'] = [-1] * config.num_tasks
    summary['acc_avg_final'] = -1
    summary['acc_avg_during'] = -1
    summary['num_weights_main'] = num
    summary['num_weights_hyper'] = hnum
    summary['num_weights_ratio'] = ratio
    summary['finished'] = 0

    shared.summary = summary

def save_summary_dict(config, shared, experiment):
    """Write a text file in the result folder that gives a quick
    overview over the results achieved so far.

    Args:
        (....): See docstring of function :func:`setup_summary_dict`.
    """
    # "setup_summary_dict" must be called first.
    assert(hasattr(shared, 'summary'))

    summary_fn = 'performance_summary.txt'
    #summary_fn = hpperm._SUMMARY_FILENAME

    with open(os.path.join(config.out_dir, summary_fn), 'w') as f:
        for k, v in shared.summary.items():
            if isinstance(v, list):
                f.write('%s %s\n' % (k, misc.list_to_str(v)))
            elif isinstance(v, float):
                f.write('%s %f\n' % (k, v))
            else:
                f.write('%s %d\n' % (k, v))

if __name__ == '__main__':
    pass
