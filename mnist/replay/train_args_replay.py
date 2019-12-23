#!/usr/bin/env python3
# Copyright 2019 Johannes von Oswald
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
# @title           :train.py
# @author          :jvo
# @contact         :voswaldj@ethz.ch
# @created         :12/16/2019
# @version         :1.0
# @python_version  :3.6.8

"""
Command-line arguments for continual learning of replay models
--------------------------------------------------------------

All command-line arguments and default values for this subpackage i.e. to 
train a MNIST replay model are handled in this module.
"""
import argparse
from datetime import datetime
import warnings

import utils.cli_args as cli

def collect_rp_cmd_arguments(mode='split', description= ""):
    """Collect command-line arguments.

    Args:
        mode: For what script should the parser assemble the set of command-line
            parameters? Options:

                - "split"
                - "perm"
    Returns:
        The Namespace object containing argument names and values.
    """
    parser = argparse.ArgumentParser(description=description)

    # If needed, add additional parameters.
    if mode == 'split':
        dout_dir = './out_split/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_argroup = cli.cl_args(parser, show_beta=False,
            show_from_scratch=True, show_multi_head=False,
            show_cl_scenario=True, show_num_tasks=True, dnum_tasks=5)
        train_argroup = cli.train_args(parser, show_lr=False, dbatch_size=128,
            dn_iter=2000, show_epochs=True)
        cli.main_net_args(parser, allowed_nets=['fc'], dfc_arch='400,400',
                  dnet_act='relu', prefix='enc_', pf_name='encoder')
        cli.main_net_args(parser, allowed_nets=['fc'], dfc_arch='400,400',
                  dnet_act='relu', prefix='dec_', pf_name='decoder')
        cli.hypernet_args(parser, dhyper_chunks=50000, dhnet_arch='10,10',
                          dtemb_size=96, demb_size=96, prefix='rp_',
                          pf_name='replay', dhnet_act='elu')
        cli.init_args(parser, custom_option=False)
        cli.miscellaneous_args(parser, big_data=False, synthetic_data=False,
            show_plots=True, no_cuda=False, dout_dir=dout_dir)
        cli.generator_args(parser, dlatent_dim=100)
        cli.eval_args(parser, dval_iter=1000)
       
        train_args_replay(parser, prefix='enc_', pf_name='encoder')
        train_args_replay(parser, show_emb_lr=True, prefix='dec_', 
                                                              pf_name='decoder')
        split_args(parser)

    elif mode == 'perm':
        dout_dir = './out_permuted/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        cl_argroup = cli.cl_args(parser, show_beta=False,
            show_from_scratch=True, show_multi_head=False,
            show_cl_scenario=True, show_num_tasks=True, dnum_tasks=10)
        train_argroup = cli.train_args(parser, show_lr=False, dbatch_size=128,
            dn_iter=5000, show_epochs=True)
        cli.main_net_args(parser, allowed_nets=['fc'], dfc_arch='1000,1000',
                  dnet_act='relu', prefix='enc_', pf_name='encoder')
        cli.main_net_args(parser, allowed_nets=['fc'], dfc_arch='1000,1000',
                  dnet_act='relu', prefix='dec_', pf_name='decoder')
        cli.hypernet_args(parser, dhyper_chunks=85000, dhnet_arch='25,25',
                          dtemb_size=24, demb_size=8, prefix='rp_',
                          pf_name='replay', dhnet_act='elu')
        cli.init_args(parser, custom_option=False)
        cli.miscellaneous_args(parser, big_data=False, synthetic_data=True,
            show_plots=True, no_cuda=False, dout_dir=dout_dir)
        cli.generator_args(parser, dlatent_dim=100)
        cli.eval_args(parser, dval_iter=1000)
        train_args_replay(parser, prefix='enc_', pf_name='Encoder', dlr=0.0001)
        train_args_replay(parser, show_emb_lr=True, prefix='dec_', dlr=0.0001,
                                              dlr_emb=0.0001, pf_name='Decoder')

        perm_args(parser)

    cl_arguments_replay(parser)
    cl_arguments_general(parser)
    data_args(parser)
    
    return parser

def parse_rp_cmd_arguments(mode='split', default=False, argv=None):
    """Parse command-line arguments.

    Args:
        See docstring of method collect_cmd_arguments.
        default (optional): If True, command-line arguments will be ignored and
            only the default values will be parsed.
        argv (optional): If provided, it will be treated as a list of command-
            line argument that is passed to the parser in place of sys.argv.
    Returns:
        The Namespace object containing argument names and values.
    """

    if mode == 'split':
        description = 'Training replay model sequentially on splitMNIST'
    elif mode == 'perm':
        description = 'Training replay model sequentially on permutedMNIST'
    else:
        raise Exception('Mode "%s" unknown.' % (mode))

    parser = collect_rp_cmd_arguments(mode=mode, description = description)

    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []

    config = parser.parse_args(args=args)

    ### Check argument values!
    cli.check_invalid_argument_usage(config)
    if mode == 'split':
        if config.num_tasks > 5:
            raise ValueError('SplitMNIST may have maximally 5 tasks.')
    
    return config

def data_args(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    an argument group for special options regarding the datasets.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Dataset options
    agroup = parser.add_argument_group('Dataset options')
    agroup.add_argument('--data_dir', type=str, default='../data/',
                         help='Directory where the data is sotred.')
    return agroup

def perm_args(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    an argument group for special options regarding the Permuted MNIST experime.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### Permuted MNIST Options
    agroup = parser.add_argument_group('Permuted MNIST Options')
    agroup.add_argument('--experiment', type=str, default="permutedMNIST",
                        help='Argument specifying the dataset used.')
    agroup.add_argument('--padding', type=int, default=2,
                        help='Padding the images with zeros for the' +
                             'permutation experiments. This is done to ' +
                             'relate to results from ' +
                             'arxiv.org/pdf/1809.10635.pdf. ' +
                             'Default: %(default)s.')
    return agroup

def split_args(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    an argument group for special options regarding the splitMNIST experiment.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### splitMNIST Options
    agroup = parser.add_argument_group('splitMNIST Options')
    agroup.add_argument('--experiment', type=str, default="splitMNIST",
                        help='Argument specifying the dataset used.')                    
    agroup.add_argument('--padding', type=int, default=0,
                        help='Padding the images with zeros for the' +
                             'permutation experiments. This is done to ' +
                             'relate to results from ' +
                             'arxiv.org/pdf/1809.10635.pdf. ' +
                             'Default: %(default)s.')
    return agroup

def train_args_replay(parser, dlr=0.001, show_emb_lr=False, 
                    dlr_emb=0.001, prefix=None, pf_name=None):
    """This is a helper method of the method parse_cmd_arguments to add
    arguments to the parser that are specific to the training of the 
    replay model.

    Arguments specified in this function:
        - `lr`
        - `emb_lr`

    Args:
        dlr: Default learning rate for the optimizer. 
        show_emb_lr: Whether the option `lr_emb` should be provided.
        dlr_emb: Default learning rate for the embedding parameters of a
            hypernetwork. Provieded if show_emb_lr is set to True.
        prefix (optional): If arguments should be instantiated with a certain
            prefix. E.g., a setup requires several main network, that may need
            different settings. For instance: prefix=:code:`prefix='gen_'`.
        pf_name (optional): A name of the type of main net for which that prefix
            is needed. For instance: prefix=:code:`'generator'`.
    Returns:
        The created argument group, in case more options should be added.
    """

    assert(prefix is None or pf_name is not None)

    heading = 'Replay network training options'

    if prefix is None:
        prefix = ''
        pf_name = ''
    else:
        heading = 'Replay network training options for %s network' % pf_name
        pf_name += ''

    # Abbreviations.
    p = prefix
    n = pf_name
    agroup = parser.add_argument_group(heading)
    agroup.add_argument('--%slr' % p, type=float, default=dlr,
                            help='Learning rate of optimizer(s) for %s ' % n +
                                 '. Default: %(default)s.')
    if show_emb_lr:
        agroup.add_argument('--%slr_emb' % p, type=float, default=dlr_emb,
                            help='Learning rate of optimizer(s) for embeddings'+
                                 ' of hypernetwork for %s ' % n +
                                 '. Default: ' + '%(default)s.')
    return agroup


def cl_arguments_general(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    arguments to the parser that are specific to the general cl setup.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
    """

    parser.add_argument('--cl_reg_batch_size', type=int, default=-1,
                         help='If not "-1", then this number will determine ' +
                              'the maximum number of previous tasks that are ' +
                              'are considered when computing the regularizer. '+
                              'Hence, if the number of previous tasks is ' +
                              'than this number, then the regularizer will be '+
                              'computed only over a random subset of previous '+
                              'tasks.')
    parser.add_argument('--no_lookahead', action='store_true',
                        help='Use a simplified version of our regularizer, ' +
                             'that doesn\'t use the theta lookahead.')
    parser.add_argument('--backprop_dt', action='store_true',
                         help='Allow backpropagation through delta theta in ' +
                              'the regularizer.')
    parser.add_argument('--use_sgd_change', action='store_true',
                         help='This argument decides how delta theta (the ' +
                              'difference of the hypernet weights when taking '+
                              'a step in optimizing the task-specific loss) ' +
                              'is computed. Note, delta theta is needed to ' +
                              'compute the CL regularizer. If this option is ' +
                              'True, then we approximate delta theta by its ' +
                              'SGD version: - alpha * grad, where alpha ' +
                              'represents the learning rate. This version is ' +
                              'computationally cheaper.')    

def cl_arguments_replay(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    arguments to the parser that are specific to the continual replay setup.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.

    Returns:
        The created argument group, in case more options should be added.
    """
    ### 
    agroup = parser.add_argument_group('Replay continual learning options.')
    agroup.add_argument('--rp_beta', type=float, default=0.01,
                         help='Trade-off for the CL regularizer for the hnet ' +
                            'in the replay model.')
    agroup.add_argument('--dont_train_rp_embeddings', action='store_true',
                         help='Train embeddings of discriminator hnet.')
    agroup.add_argument('--replay_method', type=str, default='vae',
                         help='String depicting which replay method to use.' +
                            'Options are "gan" or "vae" which is as default.')
    agroup.add_argument('--infer_task_id', action='store_true',
                        help='Train a system to infer the task id. Otherwise '+
                            'we learn a model to replay and another model '+
                            'to classifier. This is HNET+TIR else HNET+R.')
    agroup.add_argument('--single_class_replay', action='store_true',
                        help='Weather or not we want the replay moderl to '+ 
                            'learn each class in every task sequentially ' +
                            'or one task (with multiple classes) at a time. '+
                            'Note the difference to class_incremental ' +
                            'learning where the new task consists of a '+
                            'task.')
    agroup.add_argument('--embedding_reset', type=str, default='normal',
                        help='How to reset hypernet embedding after training' +
                            ' a task? Possible choices are:' +
                            '"normal" - sample from a Normal Distribution' +
                            '"old_embedding" - embedding from previous task')
    agroup.add_argument('--conditional_replay', action='store_true',
                        help='Have a task specific input to the replay model.')
    agroup.add_argument('--conditional_dim', type=int, default=100,
                        help='Specifies the dim of the task specific input.')
    agroup.add_argument('--not_conditional_hot_enc', action='store_true',
                        help='If conditions should be one-hot, if not they ' +
                            'are drawn from a Gaussian.')
    agroup.add_argument('--conditional_prior', action='store_true',
                        help='Have a task specific prior mean of the replay ' +
                            'model latent space.')
    parser.add_argument('--plot_update_steps', type=int, default=200,
                         help='How often to plot.')
    parser.add_argument('--loss_fun', type=int, default=0,
                         help='If we train a replay GAN, we can specifiy the '+
                              'GAN loss. The following options are available:'+
                              '0: Vanilla GAN (Goodfellow et al., 2014).' +
                              '1: Traditional LSGAN (Mao et al., 2018).' +
                              '2: Pearson Chi^2 LSGAN (Mao et al., 2018).' +
                              '3: Wasserstein distance(Arjovsky et al., 2017).')
    return agroup

if __name__ == '__main__':
    pass


