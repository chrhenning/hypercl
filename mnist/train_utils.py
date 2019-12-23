#!/usr/bin/env python3
# Copyright 2019 Johannes Oswald
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
# @title           :train_utils.py
# @author          :jvo
# @contact         :voswaldj@ethz.ch
# @created         :10/09/2019
# @version         :1.0
# @python_version  :3.6.8

"""
Training utilities for continual learning of classifiers
---------------------------------------------------------

A collection of helper functions to keep other scripts clean. These functions
are mainly to help create networks, build datasets and set general 
configurations for training a main network trained on some MNIST variant 
which receives its weights from a hypernetwork.   
"""

import torch
import tensorboardX
from tensorboardX import SummaryWriter
import numpy as np
import random
import os
import shutil
import pickle
import warnings

from data.special.split_mnist import get_split_MNIST_handlers
from data.special.permuted_mnist import PermutedMNISTList

from mnets.mnet_interface import MainNetInterface
from mnets.mlp import MLP

from toy_example.hyper_model import HyperNetwork
from mnist.chunked_hyper_model import ChunkedHyperNetworkHandler

import utils.misc as misc
from utils import sim_utils

def _setup_environment(config):
    """Setup the general environment for training. This incorporates:\n 
        * making computation deterministic\n 
        * creating the output folder\n 
        * selecting the torch device\n 
        * creating the Tensorboard writer\n 

    Args:
        config: Command-line arguments.

    Returns:
        (tuple): Tuple containing:
        - **device**: Torch device to be used.
        - **writer**: Tensorboard writer. Note, you still have to close the 
            writer manually!
    """

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Output folder.
    if os.path.exists(config.out_dir):
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

    ### Select torch device.
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    # keep track of device also in config
    config.no_cuda = not use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using cuda: ' + str(use_cuda))

    ### Initialize summary writer.
    # Flushes every 120 secs by default.
    # DELETEME Ensure downwards compatibility.
    if not hasattr(tensorboardX, '__version__'):
        writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'summary'))
    else:
        writer = SummaryWriter(logdir=os.path.join(config.out_dir, 'summary'))

    return device, writer

def _generate_tasks(config, steps = 2):
    """Generate a set of user defined tasks. Depending on the experiment 
    conducted, a set of splitMNIST or permutedMNIST tasks is returned.
    
    Args:
        config: Command-line arguments.
        steps: Number of classes per tasks. Only applicable for splitMNIST.
    Returns:
        data_handlers: A list of data handlers.
    """

    if config.experiment == "splitMNIST":      
        return get_split_MNIST_handlers(config.data_dir, steps = steps)
    elif config.experiment == "permutedMNIST":     
        rand = np.random.RandomState(config.data_random_seed)
        pd = config.padding*2
        permutations = [None]+[rand.permutation((28+pd)*(28+pd))
                                for _ in range(config.num_tasks - 1)]
        if config.upper_bound:
            # FIXME Due to the current implementation of the
            # `PermutedMNISTList`, which resets the batch generator everytime
            # we switch the task, we have to go for the memory inefficient
            # variant here, as this upper bound requires to build batches
            # from multiple datasets.
            # Will be fixed in the future.
            from data.special.permuted_mnist import PermutedMNIST
            return [PermutedMNIST(config.data_dir, permutation=p,
                    padding=config.padding) for p in permutations]
        else:
            return PermutedMNISTList(permutations, config.data_dir,
                padding=config.padding, show_perm_change_msg=False)
    else:
        raise ValueError('Experiment %d unknown!' % config.experiment)

def generate_classifier(config, data_handlers, device):
    """Create a classifier network. Depending on the experiment and method, 
    the method manages to build either a classifier for task inference 
    or a classifier that solves our task is build. This also implies if the
    network will receive weights from a hypernetwork or will have weights 
    on its own.
    Following important configurations will be determined in order to create
    the classifier: \n 
    * in- and output and hidden layer dimensions of the classifier. \n
    * architecture, chunk- and task-embedding details of the hypernetwork. 


    See :class:`mnets.mlp.MLP` for details on the network that will be created
        to be a classifier. 

    .. note::
        This module also handles the initialisation of the weights of either 
        the classifier or its hypernetwork. This will change in the near future.
        
    Args:
        config: Command-line arguments.
        data_handlers: List of data handlers, one for each task. Needed to
            extract the number of inputs/outputs of the main network. And to
            infer the number of tasks.
        device: Torch device.
    
    Returns: 
        (tuple): Tuple containing:
        - **net**: The classifier network.
        - **class_hnet**: (optional) The classifier's hypernetwork.
    """
    n_in = data_handlers[0].in_shape[0]
    pd = config.padding*2
    
    if config.experiment == "splitMNIST":
        n_in = n_in*n_in
    else: # permutedMNIST
        n_in = (n_in+pd)*(n_in+pd)
    
    config.input_dim = n_in
    if config.experiment == "splitMNIST":
        if config.class_incremental:
            config.out_dim = 1
        else:
            config.out_dim = 2
    else: # permutedMNIST 
        config.out_dim = 10

    if config.training_task_infer or config.class_incremental:
        # task inference network
        config.out_dim = 1

    # have all output neurons already build up for cl 2
    if config.cl_scenario != 2:
        n_out = config.out_dim*config.num_tasks
    else:
        n_out = config.out_dim

    if config.training_task_infer or config.class_incremental:
        n_out = config.num_tasks 

    # build classifier
    print('For the Classifier: ')
    class_arch = misc.str_to_ints(config.class_fc_arch)
    if config.training_with_hnet:
        no_weights = True
    else:
        no_weights = False

    net = MLP(n_in=n_in, n_out=n_out, hidden_layers=class_arch, 
                            activation_fn=misc.str_to_act(config.class_net_act), 
                            dropout_rate =config.class_dropout_rate, 
                            no_weights=no_weights).to(device)
    
    print('Constructed MLP with shapes: ', net.param_shapes)

    config.num_weights_class_net = \
                        MainNetInterface.shapes_to_num_weights(net.param_shapes)
    # build classifier hnet
    # this is set in the run method in train.py
    if config.training_with_hnet:
        
        
        class_hnet = sim_utils.get_hnet_model(config, config.num_tasks, 
                                    device, net.param_shapes, cprefix= 'class_')
        init_params = list(class_hnet.parameters())

        config.num_weights_class_hyper_net = sum(p.numel() for p in
                                    class_hnet.parameters() if p.requires_grad)
        config.compression_ratio_class = config.num_weights_class_hyper_net / \
                                                    config.num_weights_class_net
        print('Created classifier Hypernetwork with ratio: ', 
                                                config.compression_ratio_class)
        if config.compression_ratio_class > 1:
            print('Note that the compression ratio is computed compared to ' + 
                  'current target network, not might not be directly ' +
                  'comparable with the number of parameters of work we ' +
                  'compare against.')                
    else:
        class_hnet = None
        init_params = list(net.parameters())
        config.num_weights_class_hyper_net = None
        config.compression_ratio_class = None

    ### Initialize network weights.
    for W in init_params:
        if W.ndimension() == 1: # Bias vector.
            torch.nn.init.constant_(W, 0)
        else:
            torch.nn.init.xavier_uniform_(W)

    # The task embeddings are initialized differently.
    if config.training_with_hnet:
        for temb in class_hnet.get_task_embs():
            torch.nn.init.normal_(temb, mean=0., std=config.std_normal_temb)

    if hasattr(class_hnet, 'chunk_embeddings'):
        for emb in class_hnet.chunk_embeddings:
            torch.nn.init.normal_(emb, mean=0, std=config.std_normal_emb)

    if not config.training_with_hnet:
        return net
    else:
        return net, class_hnet

if __name__ == '__main__':
    pass