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
# @title           :train_replay_utils.py
# @author          :jvo
# @contact         :voswaldj@ethz.ch
# @created         :07/17/2019
# @version         :1.0
# @python_version  :3.7.3

"""
Training utilities for continual learning of replay models
----------------------------------------------------------

A collection of helper functions to keep other scripts clean. These functions
are mainly to help create networks, build datasets and set general 
configurations for training a main network trained on some MNIST replay model 
where the decoder (or generator in case of a GAN replay model) 
receives its weights from a hypernetwork.   
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

def generate_replay_networks(config, data_handlers, device, 
                           create_rp_hnet=True, only_train_replay = False):
    """Create a replay model that consists of either a encoder/decoder or
    a discriminator/generator pair. Additionally, this method manages the 
    creation of a hypernetwork for the generator/decoder. 
    Following important configurations will be determined in order to create
    the replay model: 
    * in- and output and hidden layer dimensions of the encoder/decoder. 
    * architecture, chunk- and task-embedding details of decoder's hypernetwork. 

    .. note::
        This module also handles the initialisation of the weights of either 
        the classifier or its hypernetwork. This will change in the near future.

    Args:
        config: Command-line arguments.
        data_handlers: List of data handlers, one for each task. Needed to
            extract the number of inputs/outputs of the main network. And to
            infer the number of tasks.
        device: Torch device..
        create_rp_hnet: Whether a hypernetwork for the replay should be 
            constructed. If not, the decoder/generator will have 
            trainable weights on its own.
        only_train_replay: We normally do not train on the last task since we do 
            not need to replay this last tasks data. But if we want a replay 
            method to be able to generate data from all tasks then we set this 
            option to true.

    Returns:
        (tuple): Tuple containing:

        - **enc**: Encoder/discriminator network instance.
        - **dec**: Decoder/generator networkinstance.
        - **dec_hnet**: Hypernetwork instance for the decoder/generator. This 
            return value is None if no hypernetwork should be constructed.
    """

    if config.replay_method == 'gan':
        n_out = 1
    else:
        n_out =  config.latent_dim * 2

    n_in = data_handlers[0].in_shape[0]
    pd = config.padding * 2
    if config.experiment == "splitMNIST":
        n_in = n_in*n_in
    else: # permutedMNIST
        n_in = (n_in + pd) * (n_in + pd)

    config.input_dim = n_in
    if config.experiment == "splitMNIST":
        if config.single_class_replay:
            config.out_dim = 1
        else:
            config.out_dim = 2
    else: # permutedMNIST
        config.out_dim = 10

    if config.infer_task_id:
        # task inference network
        config.out_dim = 1

    # builld encoder
    print('For the replay encoder/discriminator: ')
    enc_arch = misc.str_to_ints(config.enc_fc_arch)
    enc = MLP(n_in=n_in, n_out=n_out, hidden_layers=enc_arch, 
                            activation_fn=misc.str_to_act(config.enc_net_act), 
                            dropout_rate =config.enc_dropout_rate, 
                            no_weights=False).to(device)
    print('Constructed MLP with shapes: ', enc.param_shapes)
    init_params = list(enc.parameters())
    # builld decoder
    print('For the replay decoder/generator: ')
    dec_arch = misc.str_to_ints(config.dec_fc_arch)
    # add dimensions for conditional input
    n_out =  config.latent_dim
        
    if config.conditional_replay:
        n_out += config.conditional_dim

    dec = MLP(n_in=n_out, n_out=n_in, hidden_layers=dec_arch, 
                            activation_fn = misc.str_to_act(config.dec_net_act), 
                            use_bias=True, 
                            no_weights = config.rp_beta > 0, 
                            dropout_rate = config.dec_dropout_rate).to(device)

    print('Constructed MLP with shapes: ', dec.param_shapes)
    config.num_weights_enc = \
                        MainNetInterface.shapes_to_num_weights(enc.param_shapes)

    config.num_weights_dec = \
                        MainNetInterface.shapes_to_num_weights(dec.param_shapes)
    config.num_weights_rp_net = config.num_weights_enc + config.num_weights_dec
    # we do not need a replay model for the last task
    
    # train on last task or not
    if only_train_replay:
        subtr = 0
    else:
        subtr = 1

    num_embeddings = config.num_tasks - subtr if config.num_tasks > 1 else 1

    if config.single_class_replay :
        # we do not need a replay model for the last task
        if config.num_tasks > 1:
            num_embeddings = config.out_dim*(config.num_tasks - subtr)
        else:
            num_embeddings = config.out_dim*(config.num_tasks)   

    config.num_embeddings = num_embeddings
    # build decoder hnet
    if create_rp_hnet:
        print('For the decoder/generator hypernetwork: ')    
        d_hnet = sim_utils.get_hnet_model(config, num_embeddings, 
                            device, dec.hyper_shapes_learned, cprefix= 'rp_')

        init_params += list(d_hnet.parameters())

        config.num_weights_rp_hyper_net = sum(p.numel() for p in
                                        d_hnet.parameters() if p.requires_grad)
        config.compression_ratio_rp = config.num_weights_rp_hyper_net / \
                                                        config.num_weights_dec
        print('Created replay hypernetwork with ratio: ', 
                                                config.compression_ratio_rp)
        if config.compression_ratio_rp > 1:
            print('Note that the compression ratio is computed compared to ' + 
                  'current target network,\nthis might not be directly ' +
                  'comparable with the number of parameters of methods we ' +
                  'compare against.')   
    else:
        num_embeddings = config.num_tasks - subtr
        d_hnet = None
        init_params += list(dec.parameters())
        config.num_weights_rp_hyper_net = 0
        config.compression_ratio_rp = 0
    
    ### Initialize network weights.
    for W in init_params:
        if W.ndimension() == 1: # Bias vector.
            torch.nn.init.constant_(W, 0)
        else:
            torch.nn.init.xavier_uniform_(W)

    # The task embeddings are initialized differently.
    if create_rp_hnet:
        for temb in d_hnet.get_task_embs():
            torch.nn.init.normal_(temb, mean=0., std=config.std_normal_temb)
    
    if hasattr(d_hnet, 'chunk_embeddings'):
        for emb in d_hnet.chunk_embeddings:
            torch.nn.init.normal_(emb, mean=0, std=config.std_normal_emb)

    return enc, dec, d_hnet