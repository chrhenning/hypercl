#!/usr/bin/env python3
# Copyright 2019 Johannes von Oswald

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
# @created         :07/08/2019
# @version         :1.0
# @python_version  :3.6.8
"""

Continual learning of splitMNIST with hypernetworks.
-----------------------------------------------------

The module :mod:`mnist.train_splitMNIST` implements all training logic
for the MNIST experiments (splitMNIST, permutedMNIST).

See :ref:`README <mnist-readme-reference-label>` 
for an overview how to use this script.

"""

# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import matplotlib
matplotlib.use('Agg')

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import copy

from mnist.replay.train_gan import sample as sample_gan, train_gan_one_t
from mnist.replay.train_replay import run as replay_model
from mnist.replay.train_replay import train_vae_one_t, init_plotting_embedding
from mnist.replay.train_replay import sample as sample_vae

from mnist.train_args_default import _set_default
from mnist import train_utils
from mnist import train_args 

from mnist.plotting import _plotImages
import mnist.hp_search_splitMNIST as hpsearch

from mnets.classifier_interface import Classifier

from utils import misc
import utils.optim_step as opstep
import utils.hnet_regularizer as hreg

def _save_performance_summary(config, train_iter=None):
    """Save a summary of the test results achieved so far in a easy to parse
    file (for humans and subsequent programs).

    Args:
        config: Command-line arguments.
        train_iter:  (optional) The current training iteration. 
            Though, the results written in the file correspond have there own 
            training iteration assigned.
    """
    if train_iter is None:
        train_iter = config.n_iter

    tp = dict()

    if config.upper_bound or (config.infer_task_id and config.cl_scenario == 1):
        config.num_weights_rp_net = 0
        config.num_weights_rp_hyper_net = 0 
        config.compression_ratio_rp = 0

    tp["acc_after_list"] = misc.list_to_str(config.overall_acc_list)
    tp["acc_during_list"] = misc.list_to_str(config.during_accs_final)
    tp["acc_after_mean"] = config.acc_mean 
    tp["acc_during_mean"] = sum(config.during_accs_final)/config.num_tasks
    tp["num_weights_class_net"] = config.num_weights_class_net 
    tp["num_weights_rp_net"] = config.num_weights_rp_net 
    tp["num_weights_rp_hyper_net"] = config.num_weights_rp_hyper_net 
    tp["num_weights_class_hyper_net"] = config.num_weights_class_hyper_net 
    tp["compression_ratio_rp"] = config.compression_ratio_rp
    tp["compression_ratio_class"] = config.compression_ratio_class
    tp["overall_task_infer_accuracy_list"] = \
                       misc.list_to_str(config.overall_task_infer_accuracy_list)
                           
    tp["acc_task_infer_mean"] = config.acc_task_infer_mean 
    # Note, the keywords of this dictionary are defined by the array:
    #   hpsearch._SUMMARY_KEYWORDS
    with open(os.path.join(config.out_dir,
                           hpsearch._SUMMARY_FILENAME), 'w') as f:

        assert('num_train_iter' in hpsearch._SUMMARY_KEYWORDS)

        for kw in hpsearch._SUMMARY_KEYWORDS:
            if kw == 'num_train_iter':
                f.write('%s %d\n' % ('num_train_iter', train_iter))
                continue   
            if kw == 'finished':
                continue            
            else:
                try:
                    f.write('%s %f\n' % (kw, tp[kw]))
                except:
                    f.write('%s %s\n' % (kw, tp[kw]))

def test(dhandlers, class_nets, infer_net, device, config, writer,
                                                                task_id=None):
    """ Test continual learning experiments on MNIST dataset. This can either 
    be splitMNIST or permutedMNIST. 
    Depending on the method and cl scenario used, this methods manages
    to measure the test accuracy of a given task or all tasks after 
    training. In order to do so, correct targets need to be constructed 
    and output heads need to be set (or inferred). 
    Furthermore, this method distinguises between classification accuracy
    on a task or on the accuracy to infer task id's if applicable. 

    Args:
        (....): See docstring of function :func:`train_tasks`.
        task_id: (optional) If not None, the method will compute and return 
                   test acc for the the given task id, not all tasks.
    
    Returns:
        Scalar represting the test accuracy for the given task id.
        If ``task_id`` is None, the accuracy of the last task of the cl 
        experiment is returned. 
    """

    # get hnet if this option is given
    if class_nets is not None:
        if config.training_with_hnet:
            c_net_hnet = class_nets[1]
            c_net = class_nets[0]
            c_net.eval()
            c_net_hnet.eval()
        else:
            c_net = class_nets

    if infer_net is not None:
        infer_net.eval()

    with torch.no_grad():
         
        overall_acc = 0 
        overall_acc_list = []
        overall_task_infer_accuracy = 0
        overall_task_infer_accuracy_list = []
        
        # choose tasks to test
        if task_id is not None:
            task_range = range(task_id, task_id+1)
        else:
            task_range = range(config.num_tasks)

        # iterate through all old tasks
        for t in task_range:
            print("Testing task: ", t)
            # reset data
            if task_id is not None:
                dhandler = dhandlers[0]
            else:
                dhandler = dhandlers[t]

            # create some variables
            N_processed = 0
            test_size = dhandler.num_test_samples
            
            # is task id has to be inferred, for every x we have to do that
            # and therefore have one h(e) = W per data point - this is only 
            # possible with batch size one, for now
            if (config.infer_task_id and infer_net is not None) or \
                                                      config.infer_with_entropy:
                curr_bs = 1
            else:
                curr_bs = config.test_batch_size

            classifier_accuracy = 0
            task_infer_accuracy = 0
            Y_hat_all = []
            T_all = []

            # go through test set
            while N_processed < test_size:
                # test size of tasks might be "arbitrary"
                if N_processed + curr_bs > test_size:
                    curr_bs = test_size - N_processed
                N_processed += curr_bs

                # get data
                real_batch = dhandler.next_test_batch(curr_bs)
                X_real=dhandler.input_to_torch_tensor(real_batch[0], device, 
                    mode='inference')
                T_real=dhandler.output_to_torch_tensor(real_batch[1],device, 
                    mode='inference')

                # get short version of output dim
                od = config.out_dim
            
                #######################################
                # SET THE OUTPUT HEAD / COMPUTE TARGETS
                #######################################

                # get dummy for easy access to the  output dim of our main 
                # network as a dummy, only needed for the first iteration
                if class_nets is not None:
                    if config.training_with_hnet:
                        weights_dummy = c_net_hnet.forward(0)
                        Y_dummies = c_net.forward(X_real, weights_dummy)
                    else:
                        Y_dummies = c_net.forward(X_real)
                else:
                    Y_dummies = infer_net.forward(X_real)

                # build one hots if this option was chosen
                # here we build targets if only have one neuron per task 
                # which we set to 1
                if config.class_incremental:
                    task_out = [0, config.num_tasks]
                    T_real = torch.zeros((Y_dummies.shape[0],
                        config.num_tasks)).to(device)
                    T_real[:, t] = 1

                # compute targets - this is a bit unelegant, cl 3 requires hacks
                elif config.cl_scenario == 1 or config.cl_scenario == 2:
                    if config.cl_scenario == 1:
                        # take the task specific output neuron
                        task_out = [t*od, t*od + od]
                    else:
                        # always all output neurons (only one head is used)
                        task_out = [0, od]
                else:
                    # This here is the classic CL 3 scenario
                    # first we get the predictions, this is over all neurons
                    task_out = [0, config.num_tasks*od]
                    # Here we build the targets, this is zero everywhere 
                    # except for the current task - here the correct target
                    # is inserted

                    # build the two zero tensors that surround the targets
                    zeros1 = torch.zeros(Y_dummies[:,0:t*od].shape).\
                                                                to(device)
                    zeros2 = torch.zeros(Y_dummies[:,0:(config.num_tasks\
                                                - 1 - t)*od].shape).to(device)
                    T_real = torch.cat([zeros1, T_real, zeros2], dim = -1)

                #################
                # TASK PREDICTION
                #################

                # get task predictions
                if config.cl_scenario != 1:
                    if infer_net is not None:
                        # get infer net to predict the apparent task id 
                        task_pred = infer_net.forward(X_real)
                        task_pred = task_pred[:, 0:config.num_tasks]
                        task_pred = torch.sigmoid(task_pred)
                        _, inf_task_id = torch.max(task_pred, 1)

                        # measure acc of prediction
                        task_infer_accuracy += (inf_task_id == t).float()

                    elif config.infer_with_entropy and class_nets is not None \
                        and config.training_with_hnet:
                        entropies = []
                        if task_id is not None:
                            entrop_to_test = range(0, task_id + 1)
                        else:
                            entrop_to_test = range(config.num_tasks)
                        # infer task id through entropy of softmax outputs of 
                        # different models
                        for e in entrop_to_test:
                            weights_c = c_net_hnet.forward(e)
                            Y_hat_logits = c_net.forward(X_real, weights_c)
                            if config.cl_scenario == 2:
                                task_out = [0, od]
                            else:
                                task_out = [e*od, e*od+od]
                            Y_hat = F.softmax(Y_hat_logits[:, 
                                task_out[0]:task_out[1]]/config.soft_temp, -1)
                            entropy = -1*torch.sum(Y_hat * torch.log(Y_hat))
                            entropies.append(entropy)
                        inf_task_id = torch.argmin(torch.stack(entropies))
                        task_infer_accuracy += (inf_task_id == t).float()
                    
                    if config.cl_scenario == 3 and config.infer_output_head:
                        task_out = [inf_task_id*od, inf_task_id*od+od]
                else:
                    # if task id is known, task inference acc is 100%
                    task_infer_accuracy += 1
                    inf_task_id = t

                if class_nets is not None:
                    # from the given inf_task_id we try to produce the 
                    # correct model for that tasks
                    if config.training_with_hnet:
                        weights_c = c_net_hnet.forward(inf_task_id)
                        Y_hat_logits = c_net.forward(X_real, weights_c)
                    else:
                        Y_hat_logits = c_net.forward(X_real)

                #################
                # CLASSIFICATION
                #################
                if class_nets is not None:
                    # save predictions of current batch
                    Y_hat_logits = Y_hat_logits[:, task_out[0]:task_out[1]]
                    Y_hat = F.softmax(Y_hat_logits, dim=1)
                    if config.cl_scenario == 3 and config.infer_output_head:
                        # this is the special case where the output head is 
                        # inferred. Here we compute the argmax of the single 
                        # head and add the number of previous neurons such that
                        # it coincides with the argmax of a hot enc target   
                        # that is build for all heads. Example: we detect that
                        # task 3 is present, and every task consist of two
                        # classes. The argmax of Y_hat will either give us 0
                        # or 1, since Y_hat_logits was already cut to two 
                        # dimensions. Now we have to add 3*2 to the argmax 
                        # of Y_hat to get a prediction between class 0 and 
                        # num_tasks*class_per_task.
                        
                        Y_hat = Y_hat.argmax(dim=1, keepdim=False) + \
                                                                inf_task_id*od                                             
                    Y_hat_all.append(Y_hat)
                    T_all.append(T_real)

            if class_nets is not None:
                # append predictions
                Y_hat_all = torch.cat(Y_hat_all)
                T_all = torch.cat(T_all)
                # check if all test samples are used
                assert(Y_hat_all.shape[0] == dhandler.num_test_samples)

                # compute class acc's
                if config.cl_scenario == 3 and class_nets is not None and \
                                                       config.infer_output_head:
                    # this is a special case, we compare the
                    targets = T_all.argmax(dim=1, keepdim=False)
                    classifier_accuracy = (Y_hat_all == targets).float().mean()
                else:
                    classifier_accuracy = Classifier.accuracy(Y_hat_all, T_all)
                
                classifier_accuracy *= 100.
                print("Accuracy of task: ", t, " % "  ,  classifier_accuracy)
                overall_acc_list.append(classifier_accuracy)
                overall_acc += classifier_accuracy

            # compute task inference acc"s
            ti_accuracy=task_infer_accuracy/dhandler.num_test_samples*100.
            if config.training_task_infer or config.infer_with_entropy:
                print("Accuracy of task inference: ", t, " % "  ,  ti_accuracy)
            overall_task_infer_accuracy += ti_accuracy
            overall_task_infer_accuracy_list.append(ti_accuracy)
        
        # testing all tasks
        if task_id is None:
            if class_nets is not None:
                print("Overall mean acc: ", overall_acc/config.num_tasks)
            if config.training_task_infer or config.infer_with_entropy:
                print("Overall task inf acc: ", overall_task_infer_accuracy/ \
                                                               config.num_tasks)
            config.overall_acc_list = overall_acc_list
            config.acc_mean = overall_acc/config.num_tasks
            config.overall_task_infer_accuracy_list = \
                                                overall_task_infer_accuracy_list
            config.acc_task_infer_mean = \
                                    overall_task_infer_accuracy/config.num_tasks
            print(config.overall_task_infer_accuracy_list, config.acc_task_infer_mean)
    return classifier_accuracy

def get_fake_data_loss(dhandlers_rp, net, dec, d_hnet, device, config, writer, 
                                                                t, i, net_copy):
    """ Sample fake data from generator for tasks up to t and compute a loss
    compared to predictions of a checkpointed network.
    
    We must take caution when considering the different learning scenarios
    and methods and training stages, see detailed comments in the code.
    
    In general, we build a batch of replayed data from all previous tasks.
    Since we do not know the labels of the replayed data, we consider the
    output of the checkpointed network as ground thruth i.e. we must compute
    a loss between two logits.See :class:`mnets.classifier_interface.Classifier`
    for a detailed describtion of the different loss functions.
        
    Args:
        (....): See docstring of function :func:`train_tasks`.
        t: Task id.
        i: Current training iteration.
        net_copy: Copy/checkpoint of the classifier network before 
            learning task ``t``.
    Returns:
        The loss between predictions and predictions of a 
        checkpointed network or replayed data.
    
    """

    all_Y_hat_ls = []
    all_targets = []

    # we have to choose from which embeddings (multiple?!) to sample from 
    if config.class_incremental or config.single_class_replay:
        # if we trained every class with a different generator
        emb_num = t*config.out_dim
    else:
        # here samples from the whole task come from one generator
        emb_num = t
    # we have to choose from which embeddings to sample from 
    
    if config.fake_data_full_range:
        ran = range(0, emb_num)
        bs_per_task = int(np.ceil(config.batch_size/emb_num))
    else:
        random_t = np.random.randint(0, emb_num)
        ran = range(random_t, random_t+1)
        bs_per_task = config.batch_size

    for re in ran:

        # exchange replay data with real data to compute upper bounds 
        if config.upper_bound:
            real_batch = dhandlers_rp[re].next_train_batch(bs_per_task)
            X_fake = dhandlers_rp[re].input_to_torch_tensor(real_batch[0], 
                                                           device, mode='train')
        else:
             # get fake data
            if config.replay_method == 'gan':
                X_fake = sample_gan(dec, d_hnet, config, re, device, 
                                                               bs = bs_per_task)
            else:
                X_fake = sample_vae(dec, d_hnet, config, re, device, 
                                                               bs = bs_per_task)

        # save some fake data to the writer
        if i % 100 == 0:
            if X_fake.shape[0] >= 15:
                fig_fake = _plotImages(X_fake, config, bs_per_task)    
                writer.add_figure('train_class_' + str(re) + '_fake', 
                                                fig_fake, global_step=i)
                                            
        # compute soft targets with copied network
        target_logits = net_copy.forward(X_fake).detach()
        Y_hat_ls = net.forward(X_fake.detach())

        ###############
        # BUILD TARGETS
        ###############
        od = config.out_dim

        if config.class_incremental or config.training_task_infer:
            # This is a bit complicated: If we train class/task incrementally
            # we skip thraining the classifier on the first task. 
            # So when starting to train the classifier on task 2, we have to
            # build a hard target for this first output neuron trained by
            # replay data. A soft target (on an untrained output) would not 
            # make sense.

                # output head over all output neurons already available
            task_out = [0, (t+1)*od]
            # create target with zero everywhere except from the current re
            zeros = torch.zeros(target_logits[:,0:(t+1)*od].shape).to(device)
            
            if config.hard_targets or (t == 1 and re == 0):
                zeros[:, re] = 1
            else:
                zeros[:,0:t*od] = target_logits[:,0:t*od]

            targets = zeros
            Y_hat_ls = Y_hat_ls[:, task_out[0]:task_out[1]]

        elif config.cl_scenario == 1 or config.cl_scenario == 2:
            if config.cl_scenario == 1:
                # take the task specific output neuron
                task_out = [re*od, re*od + od]
            else:
                # always all output neurons, only one head is used
                task_out = [0, od]

            Y_hat_ls = Y_hat_ls[:, task_out[0]:task_out[1]]
            target_logits = target_logits[:, task_out[0]:task_out[1]]
            # build hard targets i.e. one hots if this option is chosen
            if config.hard_targets:
                soft_targets = torch.sigmoid(target_logits)
                zeros = torch.zeros(Y_hat_ls.shape).to(device)
                _, argmax = torch.max(soft_targets, 1)            
                targets = zeros.scatter_(1, argmax.view(-1, 1), 1)
            else:
                # loss expects logits
                targets = target_logits
        else:
            # take all neurons used up until now
           
            # output head over all output neurons already available
            task_out = [0, (t+1)*od]
            # create target with zero everywhere except from the current re
            zeros = torch.zeros(target_logits[:,0:(t+1)*od].shape).to(device)

            # sigmoid over the output head(s) from all previous task
            soft_targets = torch.sigmoid(target_logits[:,0:t*od])

            # compute one hots
            if config.hard_targets:
                _, argmax = torch.max(soft_targets, 1)
                zeros.scatter_(1, argmax.view(-1, 1), 1)
            else:
                # loss expects logits
                zeros[:,0:t*od] = target_logits[:,0:t*od]
            targets = zeros
            # choose the correct output size for the actual 
            Y_hat_ls = Y_hat_ls[:, task_out[0]:task_out[1]]
        
        # add to list
        all_targets.append(targets)
        all_Y_hat_ls.append(Y_hat_ls)
    
    # cat to one tensor
    all_targets = torch.cat(all_targets)
    Y_hat_ls = torch.cat(all_Y_hat_ls)

    if i % 200 == 0:   
        classifier_accuracy = Classifier.accuracy(Y_hat_ls, all_targets)*100.0
        msg = 'Training step {}: Classifier Accuracy: {:.3f} ' + \
            '(on current FAKE DATA training batch).'
        print(msg.format(i, classifier_accuracy))
    
    # dependent on the target softness, the loss function is chosen
    if config.hard_targets or (config.class_incremental and t == 1):
        return Classifier.logit_cross_entropy_loss(Y_hat_ls, all_targets)
    else:
        return Classifier.knowledge_distillation_loss(Y_hat_ls, all_targets)

def train_class_one_t(dhandler_class, dhandlers_rp, dec, d_hnet, net, 
                                                    device, config, writer, t):
    """Train continual learning experiments on MNIST dataset for one task.
    In this function the main training logic is implemented. 
    After setting the optimizers for the network and hypernetwork if 
    applicable, the training is structured as follows: 
    First, we get the a training batch of the current task. Depending on 
    the learning scenario, we choose output heads and build targets 
    accordingly. 
    Second, if ``t`` is greater than 1, we add a loss term concerning 
    predictions of replayed data. See :func:`get_fake_data_loss` for 
    details. Third, to protect the hypernetwork from forgetting, we add an 
    additional L2 loss term namely the difference between its current output 
    given an embedding and checkpointed targets.
    Finally, we track some training statistics.

    Args:
        (....): See docstring of function :func:`train_tasks`.
        t: Task id.
    """

    # if cl with task inference we have the classifier empowered with a hnet 
    if config.training_with_hnet:
        net_hnet = net[1]
        net = net[0]
        net.train()
        net_hnet.train()
        params_to_regularize = list(net_hnet.theta)
        optimizer = optim.Adam(params_to_regularize,
            lr=config.class_lr, betas=(0.9, 0.999))

        c_emb_optimizer = optim.Adam([net_hnet.get_task_emb(t)], 
               lr=config.class_lr_emb, betas=(0.9, 0.999))
    else:
        net.train()
        net_hnet = None
        optimizer = optim.Adam(net.parameters(),
                lr=config.class_lr, betas=(0.9, 0.999))

    # dont train the replay model if available
    if dec is not None:
        dec.eval()
    if d_hnet is not None:
        d_hnet.eval()

    # compute targets if classifier is trained with hnet
    if t > 0 and config.training_with_hnet:
        if config.online_target_computation:
            # Compute targets for the regularizer whenever they are needed.
            # -> Computationally expensive.
            targets_C = None
            prev_theta = [p.detach().clone() for p in net_hnet.theta]
            prev_task_embs = [p.detach().clone() for p in \
                                                      net_hnet.get_task_embs()]
        else:
            # Compute targets for the regularizer once and keep them all in
            # memory -> Memory expensive.
            targets_C = hreg.get_current_targets(t, net_hnet)
            prev_theta = None
            prev_task_embs = None


    dhandler_class.reset_batch_generator()

    # make copy of network
    if t >= 1:
        net_copy = copy.deepcopy(net)

    # set training_iterations if epochs are set
    if config.epochs == -1:
        training_iterations = config.n_iter
    else:
        assert(config.epochs > 0)
        training_iterations = config.epochs * \
        int(np.ceil(dhandler_class.num_train_samples / config.batch_size))

    if config.class_incremental:
        training_iterations = int(training_iterations/config.out_dim)

    # Whether we will calculate the regularizer.
    calc_reg = t > 0 and config.class_beta > 0 and config.training_with_hnet

    # set if we want the reg only computed for a subset of the  previous tasks
    if config.hnet_reg_batch_size != -1:
        hnet_reg_batch_size = config.hnet_reg_batch_size
    else:
        hnet_reg_batch_size = None
    
    for i in range(training_iterations):

        # set optimizer to zero
        optimizer.zero_grad()
        if net_hnet is not None:
            c_emb_optimizer.zero_grad()

        # Get real data
        real_batch = dhandler_class.next_train_batch(config.batch_size)
        X_real = dhandler_class.input_to_torch_tensor(real_batch[0], device, 
                                                                mode='train')
        T_real = dhandler_class.output_to_torch_tensor(real_batch[1],device, 
                                                                mode='train')
        
        if i % 100 == 0 and config.show_plots:
            fig_real = _plotImages(X_real, config)
            writer.add_figure('train_class_' + str(t) + '_real', 
                                                    fig_real, global_step=i)
        
        #################################################
        # Choosing output heads and constructing targets
        ################################################# 

        # If we train a task inference net or class incremental learning we 
        # we construct a target for every single class/task
        if config.class_incremental or config.training_task_infer:
            # in the beginning of training, we look at two output neuron
            task_out = [0, t+1]
            T_real = torch.zeros((config.batch_size, task_out[1])).to(device)
            T_real[:, task_out[1] - 1] = 1

        elif config.cl_scenario == 1 or config.cl_scenario == 2:
            if config.cl_scenario == 1:
                # take the task specific output neuron
                task_out = [t*config.out_dim, t*config.out_dim + config.out_dim]
            else:
                # always all output neurons, only one head is used
                task_out = [0, config.out_dim]
        else:
            # The number of output neurons is generic and can grow i.e. we
            # do not have to know the number of tasks before we start 
            # learning.
            if not config.infer_output_head:
                task_out = [0,(t+1)*config.out_dim]
                T_real = torch.cat((torch.zeros((config.batch_size, 
                        t * config.out_dim)).to(device), 
                        T_real), dim=1)
            # this is a special case where we will infer the task id by another 
            # neural network so we can train on the correct output head direclty
            # and use the infered output head to compute the prediction
            else:
                task_out =[t*config.out_dim, t*config.out_dim + config.out_dim]
        
        # compute loss of current data
        if config.training_with_hnet:
            weights_c = net_hnet.forward(t)
        else:
            weights_c = None

        Y_hat_logits = net.forward(X_real, weights_c)
        Y_hat_logits = Y_hat_logits[:, task_out[0]:task_out[1]]

        if config.soft_targets:
            soft_label = 0.95
            num_classes = T_real.shape[1]
            soft_targets = torch.where(T_real == 1,
                torch.Tensor([soft_label]).to(device),
                torch.Tensor([(1 - soft_label) / (num_classes-1)]).to(device))
            soft_targets = soft_targets.to(device)
            loss_task = Classifier.softmax_and_cross_entropy(Y_hat_logits,
                                                        soft_targets)
        else:
            loss_task =Classifier.softmax_and_cross_entropy(Y_hat_logits,T_real)
        
        ############################
        # compute loss for fake data
        ############################

        # Get fake data (of all tasks up until now and merge into list)
        if t >= 1 and not config.training_with_hnet:
            fake_loss = get_fake_data_loss(dhandlers_rp, net, dec,d_hnet,device, 
                                        config, writer, t, i, net_copy)
            loss_task = (1-config.l_rew)*loss_task + config.l_rew*fake_loss

        
        loss_task.backward(retain_graph=calc_reg, create_graph=calc_reg and \
                           config.backprop_dt)
     
        # compute hypernet loss and fix embedding -> change current embs
        if calc_reg:
            if config.no_lookahead:
                dTheta = None
            else:
                dTheta = opstep.calc_delta_theta(optimizer,
                    config.use_sgd_change, lr=config.class_lr,
                    detach_dt=not config.backprop_dt)          
            loss_reg = config.class_beta*hreg.calc_fix_target_reg(net_hnet, t,
                        targets=targets_C, mnet=net, dTheta=dTheta, dTembs=None,
                        prev_theta=prev_theta, prev_task_embs=prev_task_embs,
                        batch_size=hnet_reg_batch_size)
            loss_reg.backward()

        # compute backward passloss_task.backward()
        if not config.dont_train_main_model:
            optimizer.step()

        if net_hnet is not None and config.train_class_embeddings:
            c_emb_optimizer.step()

        # same stats saving
        if i % 50 == 0:
            # compute accuracies for tracking
            Y_hat_logits = net.forward(X_real, weights_c)
            Y_hat_logits = Y_hat_logits[:, task_out[0]:task_out[1]]
            Y_hat = F.softmax(Y_hat_logits, dim=1)
            classifier_accuracy = Classifier.accuracy(Y_hat, T_real) * 100.0
            writer.add_scalar('train/task_%d/class_accuracy' % t,
                                                    classifier_accuracy, i)
            writer.add_scalar('train/task_%d/loss_task' % t,
                                                    loss_task, i)
            if t >= 1 and not config.training_with_hnet:
                writer.add_scalar('train/task_%d/fake_loss' % t,
                                                    fake_loss, i)

        # plot some gradient statistics
        if i % 200 == 0:
            if not config.dont_train_main_model:
                total_norm = 0
                if config.training_with_hnet:
                    params = net_hnet.theta
                else:
                    params = net.parameters()

                for p in params:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                # TODO write gradient histograms?
                writer.add_scalar('train/task_%d/main_params_grad_norms' % t,
                                                    total_norm, i)

            if net_hnet is not None and config.train_class_embeddings:
                    total_norm = 0
                    for p in [net_hnet.get_task_emb(t)]:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    writer.add_scalar('train/task_%d/hnet_emb_grad_norms' % t,
                                                    total_norm, i)
                                                                                                 
        if i % 200 == 0:
            msg = 'Training step {}: Classifier Accuracy: {:.3f} ' + \
                '(on current training batch).'
            print(msg.format(i, classifier_accuracy))

def train_tasks(dhandlers_class, dhandlers_rp, enc, dec, d_hnet, class_net,
                                      device, config, writer, infer_net = None):
    """ Train continual learning experiments on MNIST dataset.
    This is a helper function that loops over the range of tasks and 
    iteratively starts training the classifier and the replay model 
    on new tasks. Additionally, we save the task performace just after 
    training which can later be compared to the performance after training 
    on all tasks.

    Args:
        dhandlers_class: The dataset handlers for classification.
        dhandlers_rp: The dataset handlers from the replay.
        enc: The model of the encoder network.
        dec: The model of the decoder network.
        d_hnet. The model of the decoder hyper network.
        class_net: The model of the classifier.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        writer: The tensorboard summary writer.
        infer_net: (optional) Task inference net, only used for testing.

    Returns:
        A list of test accuracies of all tasks directly after training.
    """

    print('Training MNIST (task inference) classifier ...')
    
    if not (config.upper_bound or (config.infer_task_id and 
                                                      config.cl_scenario == 1)):
            if not config.trained_replay_model:
                embd_list = init_plotting_embedding(dhandlers_rp, 
                                                        d_hnet, writer, config)

    during_accs = []
    # Begin training loop for the single tasks
    for t in range(0, config.num_tasks):
        dhandler = dhandlers_class[t]
        if class_net is not None:
            if not (config.class_incremental and t == 0):
                print("Training classifier on data handler: ", t)
                train_class_one_t(dhandler, dhandlers_rp, dec, 
                                   d_hnet, class_net, device, config, writer, t)
        else:
            if t > 0:
                print("Training task inference system on data handler: ", t)
                train_class_one_t(dhandler, dhandlers_rp, dec, 
                                   d_hnet, infer_net, device, config, writer, t)
        
        if not (t == 0 and class_net is None):
            durring_cc =  test([dhandler], class_net, infer_net, device, 
                            config,  writer, task_id=t)
            during_accs.append(durring_cc)

        if not (config.upper_bound or (config.infer_task_id and 
                                                      config.cl_scenario == 1)):
            
            if not config.trained_replay_model and t < config.num_tasks-1:
                if config.replay_method == 'gan':
                    train_gan_one_t(dhandlers_rp[t], enc, dec, d_hnet, device, 
                                                  config, writer, embd_list, t)
                else:
                    train_vae_one_t(dhandlers_rp[t], enc, dec, d_hnet, device, 
                                                  config, writer, embd_list, t)
    
    return during_accs

def run(mode='split'):  

    """ Method to start MNIST experiments. 
    Depending on the configurations, here we control the creation and 
    training of the different (replay) modules for classification or 
    task inference build out of standart neural networks and their 
    corresponding hypernetworks.

    Args:
        mode (str): Training mode defines which experiments and default values 
        are loaded. Options are splitMNIST or permutedMNIST:

                - ``split``
                - ``perm``
    """
    
    ### Get command line arguments.
    config = train_args.parse_cmd_arguments(mode=mode)
   
    assert(config.experiment == "splitMNIST" or \
                                          config.experiment == "permutedMNIST")
    if not config.dont_set_default:
        config = _set_default(config)

    if config.infer_output_head:
        assert(config.infer_task_id == True)
    
    if config.cl_scenario == 1:
        assert(config.class_incremental == False)
        assert(config.single_class_replay == False)

    if config.infer_with_entropy:
        assert(config.infer_task_id == True)
    # single class only implemented for splitMNIST
    if config.single_class_replay or config.class_incremental:
        assert(config.experiment == "splitMNIST")
    
    # check range of number of tasks
    assert(config.num_tasks > 0)
    if config.experiment == "splitMNIST":
        if config.class_incremental:
            assert(config.num_tasks <= 10)
        else:
            assert(config.num_tasks <= 5)

    # the following combination is not supported 
    if config.infer_task_id:
        assert(config.class_incremental == False)

    # enforce correct cl scenario
    if config.class_incremental:
        config.single_class_replay = 1
        config.cl_scenario = 3
        print("Attention: Cl scenario 3 is enforced!")
        steps = 1   
    else:
        steps = 2 
    
    #### Get data handlers
    dhandlers_class = train_utils._generate_tasks(config, steps)

    # decide if you want to train a replay model
    # in the case where you only want a classifier and you know the task id
    # we only train a classifier + hnet. Upper bound considers the replay case 
    # but you replay real data as if the replayu model would be "perfect".
    if config.upper_bound or (config.infer_task_id and config.cl_scenario == 1):
        train_rp = False
    else:
         train_rp = True

    ### Get replay model trained continually with hnet.
    dec, d_hnet, enc, dhandlers_rp, device, writer, config = \
                                                  replay_model(config, train_rp)
    
    # if we have a replay model trained, we now train a classifier
    # that either solves a task directly (HNET+replay) or we train a model
    # that infers the task from input.

    ###############################
    # Train task inference network
    ###############################
    
    if config.infer_task_id and not config.cl_scenario == 1 and \
                                                not config.infer_with_entropy:
        print("Training task inference model ...")
        config.trained_replay_model = False
        config.training_task_infer = True
        config.training_with_hnet = False
        ### Generate task inference network.
        infer_net = train_utils.generate_classifier(config, 
                                                    dhandlers_class, device)

        ### Train the task inference network.
        config.during_accs_inference =  train_tasks(dhandlers_class, 
                                      dhandlers_rp, enc, dec, d_hnet, None, 
                                device, config, writer, infer_net = infer_net)
        ### Test network.
        print("Testing task inference model ...")
        test(dhandlers_class, None, infer_net, device, config, writer)
        config.training_with_hnet = True
        config.trained_replay_model = True
    else:
        # if we do not train an inference network we just train a model 
        # that knows it all and not
        infer_net = None
        if config.infer_with_entropy:
            config.trained_replay_model = True
        else:
            config.trained_replay_model = False

    if config.infer_task_id:
        config.training_with_hnet = True
    else:
        config.training_with_hnet = False

    ###################
    # Train classifier
    ###################
  
    config.training_task_infer = False
    
    print("Training final classifier ...")
    ### Generate another classifier network.
    class_nets = train_utils.generate_classifier(config, 
                                                    dhandlers_class, device)
    ### Train the network.
    config.during_accs_final = train_tasks(dhandlers_class, dhandlers_rp, enc, 
                    dec, d_hnet, class_nets, device, config, writer, infer_net)

    print("Testing final classifier ...")
    ### Test network.
    test(dhandlers_class, class_nets, infer_net, device, config, writer)

    _save_performance_summary(config)
    writer.close()

    print('Program finished successfully.')
    
if __name__ == '__main__':
    run()
