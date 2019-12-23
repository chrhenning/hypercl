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
# @title           :train_args.py
# @author          :jvo
# @contact         :voswaldj@ethz.ch
# @created         :12/16/2019
# @version         :1.0
# @python_version  :3.6.8

"""
Command-line arguments for continual learning of classifiers
------------------------------------------------------------

All command-line arguments and default values for this subpackage i.e. 
mnist experiments are handled in this module.
"""
import argparse
from datetime import datetime
import warnings

import utils.cli_args as cli

from mnist.replay.train_args_replay import collect_rp_cmd_arguments, \
                                                             train_args_replay

def parse_cmd_arguments(mode='split', default=False, argv=None):
     """Parse command-line arguments.

     Args:
          mode: For what script should the parser assemble the set of 
               command-line parameters? Options:

                    - "split"
                    - "perm"

          default (optional): If True, command-line arguments will be ignored 
               and only the default values will be parsed.
          argv (optional): If provided, it will be treated as a list of command-
               line argument that is passed to the parser in place of sys.argv.

     Returns:
          The Namespace object containing argument names and values.
     """
     if mode == 'split':
          description = 'Training classifier sequentially on splitMNIST'
     elif mode == 'perm':
          description = 'Training classifier sequentially on permutedMNIST'
     else:
          raise Exception('Mode "%s" unknown.' % (mode))

     parser = collect_rp_cmd_arguments(mode=mode, description=description)

     # If needed, add additional parameters.
     if mode == 'split':
          cli.main_net_args(parser, allowed_nets=['fc'], dfc_arch='400,400',
                  dnet_act='relu', prefix='class_', pf_name='classifier')
          cli.hypernet_args(parser, dhyper_chunks=42000, dhnet_arch='10,10',
                          dtemb_size=96, demb_size=96, prefix='class_',
                          pf_name='classifier', dhnet_act='relu')
          train_args_replay(parser, show_emb_lr=True, prefix='class_',dlr=0.001,
                                        dlr_emb=0.001, pf_name='Classifier')                  
     elif mode == 'perm':
          cli.main_net_args(parser, allowed_nets=['fc'], dfc_arch='1000,1000',
                  dnet_act='relu', prefix='class_', pf_name='classifier')
          cli.hypernet_args(parser, dhyper_chunks=78000, dhnet_arch='25,25',
                          dtemb_size=24, demb_size=8, prefix='class_',
                          pf_name='classifier', dhnet_act='relu')
          train_args_replay(parser,show_emb_lr=True, prefix='class_',dlr=0.0001,
                                           dlr_emb=0.0001, pf_name='Classifier')
     cl_arguments_general(parser)
     cl_arguments_classificiation(parser)
     args = None
     if argv is not None:
          if default:
               warnings.warn('Provided "argv" will be ignored since "default" '+
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
     
def cl_arguments_general(parser):
     """This is a helper method of the method parse_cmd_arguments to add
     arguments to the parser that are specific to the general cl setup.

     Args:
          parser: Object of class :class:`argparse.ArgumentParser`.

     """

     parser.add_argument('--dont_set_default', action='store_true',
                         help='Set the following arguments to values to ' +
                              'reproduce results reported in the paper.')
     parser.add_argument('--hnet_reg_batch_size', type=int, default=-1,
                         help='Number of tasks the hnet regularisation is ' +
                              'computed for. When the number is smaller than '+
                              'the number of tasks, then a random subset is ' +
                              'used.')
                              
def cl_arguments_classificiation(parser):
     """This is a helper method of the method parse_cmd_arguments to add
     arguments to the parser that are specific to the cl setup for classifiers.

     Args:
          parser: Object of class :class:`argparse.ArgumentParser`.
     Returns:
          The Namespace object containing argument names and values.
     """
     agroup =parser.add_argument_group('Classifier continual learning options.')

     agroup.add_argument('--class_beta', type=float, default=0.01,
                         help='Trade-off for the CL regularizer for the hnet ' +
                            'in the replay model.')
     agroup.add_argument('--train_class_embeddings', action='store_true',
                         help='Train embeddings of classifier hnet.')
     agroup.add_argument('--infer_output_head', action='store_true',
                         help='Infer the output head when this option is ' +
                              'is activated and cl_scenario == 3. Otherwise '+
                              'the output head grows in size when tasks are ' +
                              'added. This option does not have an effect if '+
                              'cl_scenario != 3.')
     agroup.add_argument('--class_incremental', action='store_true',
                         help='Weather or not we want class incremental '+ 
                              ' i.e. one class at a time learning. ' +
                              'otherwise we learn one task (with multiple '+
                              'classes) at a time.')
     agroup.add_argument('--upper_bound', action='store_true',
                         help='Train the classifier with "replay" data i.e '+
                              'real data. This can be regarded an upper bound.')
     agroup.add_argument('--infer_with_entropy', action='store_true',
                         help='Infer the task id by choosing the model with ' +
                              'lowest entropy. We iterate over all tasks ' +
                              'and compare the entropies of the different'+
                              'models.')
     parser.add_argument('--soft_temp', type=float, default=1.,
                         help='Scale the softmax temperature when inferring ' +
                              'task id through the entropy.')
     agroup.add_argument('--soft_targets', action='store_true',
                        help='Use soft targets for classification in general.')
     parser.add_argument('--hard_targets', action='store_true',
                         help='Use soft or hard targets for replayed data.')
     agroup.add_argument('--dont_train_main_model', action='store_true',
                         help='Dont train the main model - this could be ' +
                              'interesting if you want to e.g. only train ' +
                              'hypernetwork embeddings.')
     agroup.add_argument('--test_batch_size', type=int, default=128,
                         help='Test batch size.')
     parser.add_argument('--fake_data_full_range', action='store_true',
                         help='Compute data over all preivous tasks.')
     parser.add_argument('--online_target_computation', action='store_true',
                        help='When using "cl_reg=0", then this option will ' +
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
     parser.add_argument('--l_rew', type=float, default=0.5,
                         help='Weight the loss between real and fake data. ' + 
                              'l_rew < 0.5, real data loss is amplified, ' +
                              'l_rew > 0.5, fake data loss is amplified. ')
     return agroup

if __name__ == '__main__':
    pass


