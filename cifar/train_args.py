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
# @title          :cifar/train_args.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :12/12/2019
# @version        :1.0
# @python_version :3.6.8
"""
Command-line arguments for CIFAR-10/100 experiments
---------------------------------------------------

The module :mod:`cifar/train_args` contains all command-line arguments and
default values for this subpackage are handled in this module.
"""
import argparse
from datetime import datetime
import warnings

import utils.cli_args as cli

def parse_cmd_arguments(mode='resnet_cifar', default=False, argv=None):
    """Parse command-line arguments.

    Args:
        mode (str): For what script should the parser assemble the set of
            command-line parameters? Options:

                - ``resnet_cifar``
                - ``zenke_cifar``

        default (bool, optional): If ``True``, command-line arguments will be
            ignored and only the default values will be parsed.
        argv (list, optional): If provided, it will be treated as a list of
            command- line argument that is passed to the parser in place of
            :code:`sys.argv`.

    Returns:
        (argparse.Namespace): The Namespace object containing argument names and
            values.
    """
    if mode == 'resnet_cifar':
        description = 'CIFAR-10/100 CL experiment using a Resnet-32'
    elif mode == 'zenke_cifar':
        description = 'CIFAR-10/100 CL experiment using the Zenkenet'
    else:
        raise ValueError('Mode "%s" unknown.' % (mode))

    parser = argparse.ArgumentParser(description=description)

    general_options(parser)

    if mode == 'resnet_cifar':
        dout_dir = './out_resnet/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_group = cli.cl_args(parser, show_beta=True, dbeta=0.05,
            show_from_scratch=True, show_multi_head=False,
            show_cl_scenario=True, show_split_head_cl3=False,
            show_num_tasks=True, dnum_tasks=6)
        cli.main_net_args(parser, allowed_nets=['resnet'], show_batchnorm=False,
            show_no_batchnorm=True, show_bn_no_running_stats=True,
            show_bn_distill_stats=True, show_bn_no_stats_checkpointing=True,
            show_specnorm=False, show_dropout_rate=False, show_net_act=False)
        cli.hypernet_args(parser, dhyper_chunks=7000, dhnet_arch='',
                          dtemb_size=32, demb_size=32)
        cli.data_args(parser, show_disable_data_augmentation=True)
        train_agroup = cli.train_args(parser, show_lr=True, dlr=0.001,
            show_epochs=True, depochs=200, dbatch_size=32,
            dn_iter=2000, show_use_adam=True, show_use_rmsprop=True,
            show_use_adadelta=False, show_use_adagrad=False,
            show_clip_grad_value=False, show_clip_grad_norm=False)

    elif mode == 'zenke_cifar':
        dout_dir = './out_zenke/run_' + \
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        cl_group = cli.cl_args(parser, show_beta=True, dbeta=0.01,
            show_from_scratch=True, show_multi_head=False,
            show_cl_scenario=True, show_split_head_cl3=False,
            show_num_tasks=True, dnum_tasks=6)
        cli.main_net_args(parser, allowed_nets=['zenke'], show_batchnorm=False,
            show_no_batchnorm=False, show_dropout_rate=True, ddropout_rate=0.25,
            show_specnorm=False, show_net_act=False)
        cli.hypernet_args(parser, dhyper_chunks=5500, dhnet_arch='100,150,200',
                          dtemb_size=48, demb_size=80)
        cli.data_args(parser, show_disable_data_augmentation=True)
        train_agroup = cli.train_args(parser, show_lr=True, dlr=0.0001,
            show_epochs=True, depochs=80, dbatch_size=256,
            dn_iter=2000, show_use_adam=True,
            dadam_beta1=0.5, show_use_rmsprop=True,
            show_use_adadelta=False, show_use_adagrad=False,
            show_clip_grad_value=False, show_clip_grad_norm=False)

    special_cl_options(cl_group)
    special_train_options(train_agroup)
    init_group = cli.init_args(parser, custom_option=True)
    special_init_options(init_group)
    cli.eval_args(parser, show_val_batch_size=True, dval_batch_size=1000)
    cli.miscellaneous_args(parser, big_data=False, synthetic_data=False,
        show_plots=False, no_cuda=False, dout_dir=dout_dir)

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

    ### ... insert additional checks if necessary
    if config.num_tasks < 1 or config.num_tasks > 11:
        raise ValueError('Argument "num_tasks" must be between 1 and 11!')

    if config.cl_scenario != 1:
        raise NotImplementedError('CIFAR experiments are currently only ' +
            'implemented for CL1.')

    if config.plateau_lr_scheduler and config.epochs == -1:
        raise ValueError('Flag "plateau_lr_scheduler" can only be used if ' +
                         '"epochs" was set.')

    if config.lambda_lr_scheduler and config.epochs == -1:
        raise ValueError('Flag "lambda_lr_scheduler" can only be used if ' +
                         '"epochs" was set.')

    if config.no_lookahead and config.backprop_dt:
        raise ValueError('Can\'t activate "no_lookahead" and "backprop_dt" ' +
                         'simultaneously.')

    return config

def general_options(parser):
    """This is a helper function of the function `parse_cmd_arguments` to create
    an argument group for general stuff important for the types of experiments
    conducted here.

    Args:
        parser (:class:`argparse.ArgumentParser`): The argument parser to which
            the argument group should be added.

    Returns:
        The created argument group, in case more options should be added.
    """
    agroup = parser.add_argument_group('General options')

    agroup.add_argument('--mnet_only', action='store_true',
                        help='Train the main network without a hypernetwork. ' +
                             'No continual learning support!')

def special_init_options(agroup):
    """This is a helper function of the function `parse_cmd_arguments` to add
    arguments to the `initialization` argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.init_args`.
    """
    agroup.add_argument('--hnet_init_shift', action='store_true',
                        help='Shift the initial hnet output such that it ' +
                             'resembles a xavier or normal init for the ' + 
                             'target network.' )

def special_cl_options(agroup):
    """This is a helper function of the function `parse_cmd_arguments` to add
    arguments to the `continual learning` argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.cl_args`.
    """
    agroup.add_argument('--init_with_prev_emb', action='store_true',
                        help='Initialize embeddings of new tasks with the ' +
                             'embedding of the most recent task.')
    agroup.add_argument('--continue_emb_training', action='store_true',
                        help='Continue the training of task embeddings for ' +
                             'old tasks. This will give further flexibility ' +
                             'to the hypernet in terms of finding a ' +
                             'configuration that preserves previous outputs ' +
                             'and generates a suitable new output.')
    agroup.add_argument('--online_target_computation', action='store_true',
                        help='For our CL regularizer, this option will ' +
                             'ensure that the targets are computed on the ' +
                             'fly, using the hypernet weights acquired after ' +
                             'learning the previous task. Note, this ' +
                             'option ensures that there is alsmost no memory ' +
                             'grow with an increasing number of tasks ' +
                             '(except from an increasing number of task ' +
                             'embeddings). If this option is deactivated, ' +
                             'the more computationally efficient way is ' +
                             'chosen of computing all main network weight ' +
                             'targets (from all previous tasks) once before ' +
                             'learning a new task.')
    agroup.add_argument('--cl_reg_batch_size', type=int, default=-1,
                        help='If not "-1", then this number will determine ' +
                             'the maximum number of previous tasks that are ' +
                             'are considered when computing the regularizer. ' +
                             'Hence, if the number of previous tasks is ' +
                             'than this number, then the regularizer will be ' +
                             'computed only over a random subset of previous ' +
                             'tasks. Default: %(default)s.')
    agroup.add_argument('--no_lookahead', action='store_true',
                        help='Use a simplified version of our regularizer, ' +
                             'that doesn\'t use the theta lookahead.')
    agroup.add_argument('--backprop_dt', action='store_true',
                        help='Allow backpropagation through "delta theta" in ' +
                             'the regularizer.')

def special_train_options(agroup):
    """This is a helper function of the function `parse_cmd_arguments` to add
    arguments to the `training` argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.train_args`.
    """
    agroup.add_argument('--plateau_lr_scheduler', action='store_true',
                        help='Will enable the usage of the learning rate ' +
                             'scheduler torch.optim.lr_scheduler.' +
                             'ReduceLROnPlateau. Note, this option requires ' +
                             'that the argument "epochs" has been set.')
    agroup.add_argument('--lambda_lr_scheduler', action='store_true',
                        help='Will enable the usage of the learning rate ' +
                             'scheduler torch.optim.lr_scheduler.' +
                             'LambdaLR. Note, this option requires ' +
                             'that the argument "epochs" has been set. ' +
                             'The scheduler will behave as specified by ' +
                             'the function "lr_schedule" in ' +
                             'https://keras.io/examples/cifar10_resnet/.')
    agroup.add_argument('--soft_targets', action='store_true',
                        help='Use soft targets for classification.')                           

if __name__ == '__main__':
    pass


