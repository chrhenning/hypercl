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
# @title           :train_args.py
# @author          :jvo
# @contact         :voswaldj@ethz.ch
# @created         :12/16/2019
# @version         :1.0
# @python_version  :3.6.8

"""
A collection of helper functions to keep other scripts clean. The main purpose
is to set the hyperparameter values to reproduce results reported in the paper. 
"""

import argparse
from datetime import datetime
from warnings import warn

def _set_default(config):
    """Overwrite default configs.

    Args:
        config: The command line arguments.
    Returns:
        Altered configs to reproduce results reported in the paper.
    """
    config.latent_dim = 100
    config.conditional_replay = True
    config.fake_data_full_range = True
    config.show_plots = True
    config.train_class_embeddings = True
    
    if config.experiment == "splitMNIST":
        config = _set_default_split(config)
    else:
        config = _set_default_permuted(config)

    config.infer_output_head = False
    if config.cl_scenario == 3 and config.infer_task_id:
        config.infer_output_head = True
    
    if config.replay_method == "gan":
        config = _set_default_gan(config)
        
    return config

def _set_default_split(config):
    """Overwrite default configs for splitMNIST.

    Args:
        config: The command line arguments.
    Returns:
        Altered configs to reproduce results reported in the paper.
    """

    # General setup
    config.enc_fc_arch = '400,400'
    config.dec_fc_arch = '400,400'
    config.class_fc_arch = '400,400'
    config.enc_lr, config.dec_lr, config.dec_lr_emb = 0.001, 0.001, 0.001
    config.class_lr, config.class_lr_emb = 0.001, 0.001
    config.n_iter = 2000
    config.batch_size = 128
    config.data_dir = '../datasets'
    config.num_tasks = 5
    config.padding = 0
    config.no_lookahead = False
    config.hard_targets = False

    # VAE hnet
    config.rp_temb_size = 96
    config.rp_emb_size = 96
    config.rp_hnet_act = "elu"
    config.rp_hyper_chunks = 50000
    config.rp_hnet_arch = '10,10'
    config.rp_beta = 0.01

    # Classifier hnet
    config.class_temb_size = 96
    config.class_emb_size = 96
    config.class_hnet_act = "relu"
    config.class_hyper_chunks = 42000
    config.class_hnet_arch = '10,10'
    config.class_beta = 0.01

    if config.infer_task_id:
        #HNET+TIR
        config.hard_targets = True
        config.dec_fc_arch = '300,150'
        config.rp_beta = 0.05
    else:
        #HNET+R
        config.dec_fc_arch = '250,350'
        
    return config

def _set_default_permuted(config):
    """Overwrite default configs for permutedMNIST.

    Args:
        config: The command line arguments.
    Returns:
        Altered configs to reproduce results reported in the paper.
    """
    if config.num_tasks < 10:
        warn('Training permuted with num tasks = %d. ' % (config.num_tasks))
    
    # General setup
    config.enc_fc_arch = '1000,1000'
    config.dec_fc_arch = '1000,1000'
    config.class_fc_arch = '1000,1000'
    config.enc_lr, config.dec_lr, config.dec_lr_emb = 0.0001, 0.0001, 0.0001
    config.class_lr, config.class_lr_emb = 0.0001, 0.0001
    config.n_iter = 5000
    config.batch_size = 128
    config.data_dir = '../datasets'
    config.padding = 2
    config.no_lookahead = False

    # VAE hnet
    config.rp_temb_size = 24
    config.rp_emb_size = 8
    config.rp_hnet_act = "elu"
    config.rp_hyper_chunks = 85000
    config.rp_hnet_arch = '25,25'
    config.rp_beta = 0.1

    # Classifier hnet
    config.class_temb_size = 24
    config.class_emb_size = 8
    config.class_hnet_act = "elu"
    config.class_hyper_chunks = 78000
    config.class_hnet_arch = '25,25'
    config.class_beta = 0.1
    
    # small capacity
    if config.class_fc_arch == '100,100':
        # Classifier hnet
        config.class_temb_size = 64
        config.class_emb_size = 12
        config.class_hyper_chunks = 2000
        config.class_beta = 0.05
        config.class_hnet_arch = '100,75,50'

    #perm100 classifier config
    if config.num_tasks >= 100:
        print("Attention: permuted100 not tested after the code migration.")
        config.class_temb_size = 128
        config.class_emb_size = 12
        config.class_hnet_act = "relu"
        config.class_hnet_arch = '200,250,350'
        config.class_hyper_chunks = 7500
        config.class_beta = 0.01

    #perm250
    if config.num_tasks >= 250:
        print("Attention: permuted250 not tested after the code migration.")
        config.class_hyper_chunks = 12000
        config.hnet_reg_batch_size = 32
        config.online_target_computation = True

    #HNET+TIR
    if config.infer_task_id:
        config.hard_targets = True
        #perm100 replay config
        if config.num_tasks >= 100:
            config.dec_fc_arch = '400,600'
            config.rp_hnet_arch = '100'
            config.rp_hyper_chunks = 20000
            config.rp_temb_size = 128
            config.rp_emb_size = 12
            if config.cl_scenario == 2:
                config.hnet_reg_batch_size = 20
    #HNET+R
    else:
        config.dec_fc_arch = '400,400'
        config.hard_targets = False

    return config

def _set_default_gan(config):
    """Overwrite default configs for GAN training.

    Args:
        config: The command line arguments.
    Returns:
        Altered configs to reproduce results reported in the paper.
    """

    #TODO
    
    return config

if __name__ == '__main__':
    pass


