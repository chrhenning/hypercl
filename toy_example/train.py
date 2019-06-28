#!/usr/bin/env python3
# Copyright 2018 Christian Henning
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
@title           :toy_regression/train.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :10/22/2018
@version         :1.0
@python_version  :3.6.6

Train a toy regression task using two auxiliarry networks. One (the Hypernet)
will be generating the weights for the main network. The other (the
Discriminator) is going to decide whether the main network generated meaningful
weights.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
from warnings import warn

from toy_example import train_utils
from toy_example.task_recognition_model import RecognitionNet
from utils import ewc_regularizer as ewc
from toy_example.main_model import MainNetwork

import utils.optim_step as opstep
import utils.hnet_regularizer as hreg

def test(data_handlers, mnet, hnet, device, config, writer, rnet=None, \
         immediate_mse=None, immediate_weights=None, baseline_mse=None,
         save_fig=True):
    """Test the performance of all tasks.

    Args:
        See docstring of method train_reg. Note, "hnet" can be passed as None.
        In this case, no weights are passed to the "forward" method of the main
        network.
        data_handlers: A list of data handlers, each representing a task.
        rnet (optional): The task recognition network. If given, the of each
            sample will be inferred via this network.
        immediate_mse (optional): A list of floats with the same length as
            "data_handlers". Each number in the list will be considered as the
            immeaiate MSE achieved after training on the corresponding task.
            Entries with value "-1." are replaced by the current MSE (hence, 
            these would be the tasks that just have been trained).
        immediate_weights (optional): A list of 1D vectos, each denoting the
            flattened output of the hypernetwork for a task immediatelty after
            training. Again, entries with value "-1" are replaced by the current
            hypernet output.
        baseline_mse (optional): A dictionary of labels mapping to float arrays.
            These will be MSE values that are added to the MSE plot. Note, it is
            the responsibility of the user of this argument to ensure that these
            baseline values are comparible to the current training setting (in
            terms of hyperparameters like network size or training schedule).
        save_fig: Whether the figures should be saved in the output folder.

    Returns:
        mse: The current MSE values.
        immediate_mse: The given "immediate_mse", potentially modified by
            replacing missing values with current ones.
        immediate_weights: The given "immediate_weights", potentially modified
            by replacing missing values with current ones.
    """
    assert(hnet is not None or immediate_weights is None)
    # Recognition model only makes sense if a hypernetwork is used.
    assert(rnet is None or hnet is not None)

    print('### Testing all trained tasks ... ###')

    mnet.eval()
    if hnet is not None:
        hnet.eval()

    with torch.no_grad():
        n = len(data_handlers)
        mse = np.zeros(n)

        inputs = []
        predictions = []
        if rnet is not None:
            accuracies = np.zeros(n)
            num_correct = 0
            num_total = 0

        for i in range(n):
            data = data_handlers[i]

            X = data.get_test_inputs()
            T = data.get_test_outputs()

            X = data.input_to_torch_tensor(X, device)
            T = data.output_to_torch_tensor(T, device)

            if hnet is None:
                Y = mnet.forward(X)
            else:
                weights = hnet.forward(i)
                if rnet is None:
                    Y = mnet.forward(X, weights)
                else:
                    Y = torch.empty_like(T)
                    preds = torch.empty((X.shape[0]), dtype=torch.int64)

                    for s in range(X.shape[0]):
                        x_sample = X[s,:].view(1, -1)
                        if isinstance(rnet, RecognitionNet):
                            alpha, _, _ = rnet.encode(x_sample)
                        else: # MainModel
                            alpha, _ = rnet.forward(x_sample)
                        preds[s] = alpha.argmax(dim=1, keepdim=False)

                        W = hnet.forward(preds[s])

                        Y[s,:] = mnet.forward(x_sample, W)

                    num_total += X.shape[0]
                    num_correct += (preds == i).sum().cpu().numpy()

                    acc = (preds == i).float().mean() * 100.
                    accuracies[i] = acc.cpu().numpy()

            if config.multi_head:
                n_y = data.out_shape[0]
                allowed_outputs = list(range(i*n_y, (i+1)*n_y))
                Y = Y[:, allowed_outputs]

            mse[i] = F.mse_loss(Y, T)

            inputs.append(X.data.cpu().numpy())
            predictions.append(Y.data.cpu().numpy())

            if immediate_mse is not None and immediate_mse[i] == -1:
                immediate_mse[i] = mse[i]

            if immediate_weights is not None:
                W_curr = torch.cat([d.clone().view(-1) for d in weights])
                if type(immediate_weights[i]) == float and \
                        immediate_weights[i] == -1:
                    immediate_weights[i] = W_curr
                else:
                    W_immediate = immediate_weights[i]
                    W_dis = torch.norm(W_curr - W_immediate, 2)
                    print('Euclidean distance between hypernet output for ' +
                          'task %d: %f' % (i, W_dis))

    fig_fn = None
    if save_fig:
        fig_fn = os.path.join(config.out_dir, '%d_tasks_plot' % n)
    data_handlers[0].plot_datasets(data_handlers, inputs, predictions,
                                   filename=fig_fn, figsize=None)

    plt.figure()
    plt.title('Current MSE on each task.')
    plt.scatter(np.arange(1, n+1), mse, label='Current MSE')
    if immediate_mse is not None:
        plt.scatter(np.arange(1, n+1), immediate_mse[:n], label='Immediate MSE',
                    marker='x')
    if baseline_mse is not None:
        for k, v in baseline_mse.items():
            plt.scatter(np.arange(1, n + 1), v[:n], label=k, marker='+')
    plt.ylabel('MSE')
    plt.xlabel('Task')
    plt.xticks(np.arange(1, n+1))
    plt.legend()
    if save_fig:
        plt.savefig(os.path.join(config.out_dir, '%d_tasks_mse' % n),
                    bbox_inches='tight')
    plt.show()

    print('Mean task MSE: %f (std: %d)' % (mse.mean(), mse.std()))

    if rnet is not None:
        plt.figure()
        plt.title('Current recognition accuracy on each task.')
        plt.scatter(np.arange(1, n + 1), accuracies)
        plt.ylabel('Accuracy')
        plt.xlabel('Task')
        plt.xticks(np.arange(1, n + 1))
        plt.show()

        print('Overall accuracy of recognition model: %.2f%%.' %
              (100. * num_correct / num_total))

    print('### Testing all trained tasks ... Done ###')

    return mse, immediate_mse, immediate_weights

def evaluate(task_id, data, mnet, hnet, device, config, writer,
             train_iter=None):
    """Evaluate the network. Evaluate the performance of the network on a
    single task on the validation set.

    Note, if no validation set is available, the test set will be used instead.

    Args:
        See docstring of method train_reg. Note, "hnet" can be passed as None.
        In this case, no weights are passed to the "forward" method of the main
        network.
        train_iter: The current training iteration.
    """
    if train_iter is None:
        print('### Final evaluation ...')
    else:
        print('# Evaluating network on task %d before running training step %d '
              % (task_id, train_iter) + '...')

    mnet.eval()
    if hnet is not None:
        hnet.eval()

    with torch.no_grad():
        if data.num_val_samples == 0:
            X = data.get_test_inputs()
            T = data.get_test_outputs()
        else:
            X = data.get_val_inputs()
            T = data.get_val_outputs()

        X = data.input_to_torch_tensor(X, device)
        T = data.output_to_torch_tensor(T, device)

        if hnet is None:
            Y = mnet.forward(X)
        else:
            weights = hnet.forward(task_id)
            Y = mnet.forward(X, weights)

        if config.multi_head:
            n_y = data.out_shape[0]
            allowed_outputs = list(range(task_id*n_y, (task_id+1)*n_y))
            Y = Y[:, allowed_outputs]

        mse = F.mse_loss(Y, T)

        print('Eval - MSE loss: %f.' % (mse))

        #data.plot_samples('Evaluation at train iter %d' % train_iter,
        #                  X.data.cpu().numpy(), outputs=T.data.cpu().numpy(),
        #                  predictions=Y.data.cpu().numpy())
        data.plot_predictions([X.data.cpu().numpy(), Y.data.cpu().numpy()])

def evaluate_rnet(task_id, data, rnet, device, config, writer, train_iter):
    """Evaluate the performance of the recognition model during training on a
    single task.

    Note, if no validation set is available, the test set will be used instead.

    Args:
        See docstring of method train_reg.
        rnet: The recognition model.
        train_iter: The current training iteration.
    """
    print('# Evaluating the recognition model on task %d before ' % (task_id) +
          'running training step %d ...' % (train_iter))

    rnet.eval()

    with torch.no_grad():
        if data.num_val_samples == 0:
            X = data.get_test_inputs()
        else:
            X = data.get_val_inputs()

        X = data.input_to_torch_tensor(X, device)
        # Task recognition targets:
        T = torch.ones(X.shape[0], dtype=torch.int64).to(device) * task_id

        alpha, _, z = rnet.encode(X)
        X_rec = rnet.decode(alpha, z)

        task_preds = alpha.argmax(dim=1, keepdim=False)
        acc = (task_preds == T).float().mean() * 100.

        mse = rnet.reconstruction_loss(X, X_rec)

        print('Eval - recognition model: Reconstruction %.4f (MSE), ' % (mse) +
              'Task Recognition %.2f%% (Accuracy)' % (acc))

        writer.add_scalar('val_ae/task_%d/mse' % task_id, mse, train_iter)
        writer.add_scalar('val_ae/task_%d/accuracy' % task_id, acc, train_iter)

def calc_reg_masks(data_handlers, mnet, device, config):
    """Compute the regularizer mask for each task when using a multi-head setup.
    See method "get_reg_masks" of class "MainNetwork" for more details.

    Deprecated: Method "calc_fix_target_reg" has its own way of handling
    multihead setups that is more memory efficient than keeping a mask for each
    task.

    Args:
        See docstring of method train_reg.
        data_handlers: A list of all data_handlers, one for each task.

    Returns:
        A list of regularizer masks.
    """
    assert(config.multi_head and config.masked_reg)

    warn('"calc_reg_masks" is deprecated and not maintained as it is unused ' +
         'currently.', DeprecationWarning)

    assert(mnet.has_fc_out)
    main_shapes = mnet.hyper_shapes

    masks = []
    for i, data in enumerate(data_handlers):
        n_y = data.out_shape[0]
        allowed_outputs = list(range(i*n_y, (i+1)*n_y))

        masks.append(MainNetwork.get_reg_masks(main_shapes, allowed_outputs,
                                               device, use_bias=mnet.has_bias))

    return masks

def train_reg(task_id, data, mnet, hnet, device, config, writer):
    """Train the network using the task-specific loss plus a regularizer that
    should weaken catastrophic forgetting.

    loss = task_loss + beta * regularizer.

    Args:
        task_id: The index of the task on which we train.
        data: The dataset handler.
        mnet: The model of the main network.
        hnet: The model of the hyoer network.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        writer: The tensorboard summary writer.
    """
    print('Training network ...')

    mnet.train()
    hnet.train()

    regged_outputs = None
    if config.multi_head:
        n_y = data.out_shape[0]
        out_head_inds = [list(range(i*n_y, (i+1)*n_y)) for i in
                         range(task_id+1)]
        # Outputs to be regularized.
        regged_outputs = out_head_inds if config.masked_reg else None
    allowed_outputs = out_head_inds[task_id] if config.multi_head else None

    # Collect Fisher estimates for the reg computation.
    fisher_ests = None
    if config.ewc_weight_importance and task_id > 0:
        fisher_ests = []
        n_W = len(hnet.target_shapes)
        for t in range(task_id):
            ff = []
            for i in range(n_W):
                _, buff_f_name = ewc._ewc_buffer_names(t, i, False)
                ff.append(getattr(mnet, buff_f_name))
            fisher_ests.append(ff)

    # Regularizer targets.
    if config.reg == 0:
        targets = hreg.get_current_targets(task_id, hnet)

    regularized_params = list(hnet.theta)
    if task_id > 0 and config.plastic_prev_tembs:
        assert(config.reg == 0)
        for i in range(task_id): # for all previous task embeddings
            regularized_params.append(hnet.get_task_emb(i))
    theta_optimizer = optim.Adam(regularized_params, lr=config.lr_hyper)
    # We only optimize the task embedding corresponding to the current task,
    # the remaining ones stay constant.
    emb_optimizer = optim.Adam([hnet.get_task_emb(task_id)],
                               lr=config.lr_hyper)

    # Whether the regularizer will be computed during training?
    calc_reg = task_id > 0 and config.beta > 0

    for i in range(config.n_iter):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            evaluate(task_id, data, mnet, hnet, device, config, writer, i)
            mnet.train()
            hnet.train()

        if i % 100 == 0:
            print('Training iteration: %d.' % i)

        ### Train theta and task embedding.
        theta_optimizer.zero_grad()
        emb_optimizer.zero_grad()

        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')

        weights = hnet.forward(task_id)
        Y = mnet.forward(X, weights)
        if config.multi_head:
            Y = Y[:, allowed_outputs]

        # Task-specific loss.
        loss_task = F.mse_loss(Y, T)
        # We already compute the gradients, to then be able to compute delta
        # theta.
        loss_task.backward(retain_graph=calc_reg,
                           create_graph=config.backprop_dt and calc_reg)

        # The task embedding is only trained on the task-specific loss.
        # Note, the gradients accumulated so far are from "loss_task".
        emb_optimizer.step()

        # DELETEME check correctness of opstep.calc_delta_theta.
        #dPrev = torch.cat([d.data.clone().view(-1) for d in hnet.theta])
        #dT_estimate = torch.cat([d.view(-1).clone() for d in
        #    opstep.calc_delta_theta(theta_optimizer,
        #                            config.use_sgd_change, lr=config.lr_hyper,
        #                            detach_dt=not config.backprop_dt)])

        loss_reg = 0
        dTheta = None
        grad_tloss = None
        if calc_reg:
            if i % 100 == 0:  # Just for debugging: displaying grad magnitude.
                grad_tloss = torch.cat([d.grad.clone().view(-1) for d in
                                        hnet.theta])

            dTheta = opstep.calc_delta_theta(theta_optimizer,
                config.use_sgd_change, lr=config.lr_hyper,
                detach_dt=not config.backprop_dt)
            if config.plastic_prev_tembs:
                dTembs = dTheta[-task_id:]
                dTheta = dTheta[:-task_id]
            else:
                dTembs = None

            if config.reg == 0:
                loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                    targets=targets, dTheta=dTheta, dTembs=dTembs, mnet=mnet,
                    inds_of_out_heads=regged_outputs,
                    fisher_estimates=fisher_ests)
            elif config.reg == 1:
                loss_reg = hreg.calc_value_preserving_reg(hnet, task_id, dTheta)
            elif config.reg == 2:
                loss_reg = hreg.calc_jac_reguarizer(hnet, task_id, dTheta,
                                                    device)
            elif config.reg == 3: # EWC
                loss_reg = ewc.ewc_regularizer(task_id, hnet.theta, None,
                    hnet=hnet, online=config.online_ewc, gamma=config.gamma)
            loss_reg *= config.beta

            loss_reg.backward()

            if grad_tloss is not None:
                grad_full = torch.cat([d.grad.view(-1) for d in hnet.theta])
                # Grad of regularizer.
                grad_diff = grad_full - grad_tloss
                grad_diff_norm = torch.norm(grad_diff, 2)
                
                # Cosine between regularizer gradient and task-specific
                # gradient.
                dT_vec = torch.cat([d.view(-1).clone() for d in dTheta])
                grad_cos = F.cosine_similarity(grad_diff.view(1,-1),
                                               dT_vec.view(1,-1))

        theta_optimizer.step()

        # DELETEME
        #dCurr = torch.cat([d.data.view(-1) for d in hnet.theta])
        #dT_actual = dCurr - dPrev
        #print(torch.norm(dT_estimate - dT_actual, 2))

        if i % 10 == 0:
            writer.add_scalar('train/task_%d/mse_loss' % task_id, loss_task, i)
            writer.add_scalar('train/task_%d/regularizer' % task_id, loss_reg,
                              i)
            writer.add_scalar('train/task_%d/full_loss' % task_id, loss_task +
                              loss_reg, i)
            if dTheta is not None:
                dT_norm = torch.norm(torch.cat([d.view(-1) for d in dTheta]), 2)
                writer.add_scalar('train/task_%d/dTheta_norm' % task_id,
                                  dT_norm, i)
            if grad_tloss is not None:
                writer.add_scalar('train/task_%d/full_grad_norm' % task_id,
                                  torch.norm(grad_full, 2), i)
                writer.add_scalar('train/task_%d/reg_grad_norm' % task_id,
                                  grad_diff_norm, i)
                writer.add_scalar('train/task_%d/cosine_task_reg' % task_id,
                                  grad_cos, i)

    if config.reg == 3:
        ## Estimate diagonal Fisher elements.
        ewc.compute_fisher(task_id, data, hnet.theta, device, mnet, hnet=hnet,
            empirical_fisher=True, online=config.online_ewc, gamma=config.gamma,
            n_max=config.n_fisher, regression=True,
            allowed_outputs=allowed_outputs)

    if config.ewc_weight_importance:
        ## Estimate Fisher for outputs of the hypernetwork.
        weights = hnet.forward(task_id)

        # Note, there are actually no parameters in the main network.
        fake_main_params = nn.ParameterList()
        for i, W in enumerate(weights):
            fake_main_params.append(nn.Parameter(torch.Tensor(*W.shape),
                                                 requires_grad=True))
            fake_main_params[i].data = weights[i]

        ewc.compute_fisher(task_id, data, fake_main_params, device, mnet,
            empirical_fisher=True, online=False, n_max=config.n_fisher,
            regression=True, allowed_outputs=allowed_outputs)

    print('Training network ... Done')

def train_proximal(task_id, data, mnet, hnet, device, config, writer):
    """Train the hypernetwork via a proximal algorithm. Hence, we don't optimize
    the weights of the hypernetwork directly (except for the task embeddings).
    Instead, we optimize the following loss for dTheta. After a few optimization
    steps, dTheta will be added to the current set of weights in the
    hypernetwork.

    loss = task_loss(theta + dTheta) + alpha ||dTheta||^2 + beta * 
           sum_{j < task_id} || h(c_j, theta) - h(c_j, theta + dTheta) ||^2.

    Args:
        See docstring of method train_reg.
    """
    if config.reg == 3 or config.ewc_weight_importance:
        # TODO Don't just copy all the code, find a more elegant solution.
        raise NotImplementedError('Chosen regularizer not implemented for ' +
                                  'proximal algorithm!')
    if config.plastic_prev_tembs:
        # TODO can be implemented as above.
        raise NotImplementedError('Option "plastic_prev_tembs" not yet ' +
                                  'implemented for proximal algorithm.')

    print('Training network ...')

    mnet.train()
    hnet.train()

    regged_outputs = None
    if config.multi_head:
        n_y = data.out_shape[0]
        out_head_inds = [list(range(i*n_y, (i+1)*n_y)) for i in
                         range(task_id+1)]
        # Outputs to be regularized.
        regged_outputs = out_head_inds if config.masked_reg else None
    allowed_outputs = out_head_inds[task_id] if config.multi_head else None

    # Regularizer targets.
    if config.reg == 0:
        targets = hreg.get_current_targets(task_id, hnet)

    # Generate dTheta
    dTheta = nn.ParameterList()
    for tshape in hnet.theta_shapes:
        dTheta.append(nn.Parameter(torch.Tensor(*tshape),
                                   requires_grad=True))
    dTheta = dTheta.to(device)

    # Initialize dTheta
    for dt in dTheta:
        dt.data.zero_()

    dtheta_optimizer = optim.Adam(dTheta, lr=config.lr_hyper)
    # We only optimize the task embedding corresponding to the current task,
    # the remaining ones stay constant.
    emb_optimizer = optim.Adam([hnet.get_task_emb(task_id)],
                               lr=config.lr_hyper)

    for i in range(config.n_iter):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            evaluate(task_id, data, mnet, hnet, device, config, writer, i)
            mnet.train()
            hnet.train()

        if i % 100 == 0:
            print('Training iteration: %d.' % i)

        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')

        ### Train theta.
        # Initialize dTheta
        # n_steps has to be high, if we don't do this reset.
        for dt in dTheta:
            dt.data.zero_()

        # Reset optimizer state in every new iteration:
        # This only seems to hurt, even if dTheta is reset to zero every
        # training iteration.
        #dtheta_optimizer = optim.Adam(dTheta, lr=config.lr_hyper)

        # Train dTheta
        dT_loss_vals = [] # For visualizations.
        for n in range(config.n_steps):
            dtheta_optimizer.zero_grad()

            weights = hnet.forward(task_id, dTheta=dTheta)
            Y = mnet.forward(X, weights)
            if config.multi_head:
                Y = Y[:, allowed_outputs]

            # Task-specific loss.
            loss_task = F.mse_loss(Y, T)

            # L2 reg for dTheta
            dTheta_norm = torch.norm(torch.cat([d.view(-1) for d in dTheta]))
            l2_reg = dTheta_norm

            # Continual learning regularizer.
            cl_reg = torch.zeros(()) # Scalar 0
            if task_id > 0 and config.beta > 0:
                if config.reg == 0:
                    cl_reg = hreg.calc_fix_target_reg(hnet, task_id,
                        targets=targets, dTheta=dTheta, mnet=mnet,
                        inds_of_out_heads=regged_outputs)
                elif config.reg == 1:
                    cl_reg = hreg.calc_value_preserving_reg(hnet, task_id,
                                                            dTheta)
                elif config.reg == 2:
                    cl_reg = hreg.calc_jac_reguarizer(hnet, task_id, dTheta,
                                                      device)

            loss = loss_task + config.alpha * l2_reg + config.beta * cl_reg
            loss.backward()
            dtheta_optimizer.step()

            dT_loss_vals.append([l.data.cpu().numpy() for l in
                                 [loss_task, l2_reg, cl_reg, loss]])

        # Apply dTheta.
        for tind, t in enumerate(hnet.theta):
            t.data = t.data + dTheta[tind].data

        ### Train class embedding.
        emb_optimizer.zero_grad()

        weights = hnet.forward(task_id)
        Y = mnet.forward(X, weights)
        if config.multi_head:
            Y = Y[:, allowed_outputs]

        loss_mse = F.mse_loss(Y, T)
        loss_mse.backward()
        emb_optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('train/task_%d/mse_loss' % task_id, loss_mse, i)
            dT_norm = torch.norm(torch.cat([d.view(-1) for d in dTheta]))
            writer.add_scalar('train/task_%d/dT_norm' % task_id, dT_norm, i)

            # We visualize the evolution of dTheta learning by looking at
            # individual timesteps (because I don't know how to visualize
            # the evolution of sequences over time in Tensorboard).
            if config.n_steps == 1:
                inds = [0]
            elif config.n_steps == 2:
                inds = [0, config.n_steps-1]
            else:
                inds = [0, config.n_steps // 2, config.n_steps-1]

            for ii in inds:
                ltask, ll2, lcl, l = dT_loss_vals[ii]
                writer.add_scalar('train/task_%d/dT_step_%d/mse' % \
                                  (task_id, ii), ltask, i)
                writer.add_scalar('train/task_%d/dT_step_%d/dT_l2_reg' % \
                                  (task_id, ii), ll2, i)
                writer.add_scalar('train/task_%d/dT_step_%d/dT_cl_reg' % \
                                  (task_id, ii), lcl, i)
                writer.add_scalar('train/task_%d/dT_step_%d/dT_full_loss' % \
                                  (task_id, ii), l, i)

    print('Training network ... Done')

def train_rnet(task_id, data, rnet, device, config, writer):
    """Train the recognition network. This means, that the encoder should be
    able to detect the current task from input samples. Though, to avoid
    forgetting (i.e., the ability to recognize previous tasks), we also need to
    ensure that we have a generative model for samples from previous tasks. This
    generative model will be the decoder state before training starts. This
    generator is used to produce batches of fake samples for each previous
    task. Hence, "we have data for all tasks at once" to train the full auto-
    encoder (task identification of all tasks inclusively).

    loss = reconstruction loss + task identification loss + prior matching
         = ||x - x_rec||^2 + beta_ce * cross_entropy + beta_pm * prior-matching

    Args:
        See docstring of method train_reg.
        rnet: The recognition network.
    """
    print('Training recognition network ...')

    rnet.train()

    # Weights of replay model.
    replay_weights = [p.detach().clone() for p in rnet.decoder_weights]

    optimizer = optim.Adam(rnet.parameters(), lr=config.lr_ae)

    for i in range(config.n_iter_ae):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            evaluate_rnet(task_id, data, rnet, device, config, writer, i)
            rnet.train()

        if i % 100 == 0:
            print('Training iteration: %d.' % i)

        ### Train theta.
        optimizer.zero_grad()

        loss_rec = 0
        loss_ce = 0
        loss_pm = 0

        for t in range(task_id+1):
            # Task recognition targets:
            T = torch.ones(config.batch_size, dtype=torch.int64).to(device) * t

            if t == task_id:
                batch = data.next_train_batch(config.batch_size)
                X = data.input_to_torch_tensor(batch[0], device, mode='train')
            else: # Generate fake data.
                # We simply use samples from the prior as possible latent space
                # realizations for the current task.
                z_fake = rnet.prior_samples(config.batch_size).to(device)
                # Fake-softmax activations are 1-hot encodings for the current
                # task "t".
                alpha_fake = torch.zeros((config.batch_size, rnet.dim_alpha)). \
                    to(device)
                alpha_fake[:, t] = 1.

                X = rnet.decode(alpha_fake, z_fake,
                    decoder_weights=replay_weights).detach().clone()

            alpha, nu_z, z, log_alpha = rnet.encode(X, ret_log_alpha=True)
            X_rec = rnet.decode(alpha, z)

            loss_rec += rnet.reconstruction_loss(X, X_rec)
            loss_ce += rnet.task_cross_entropy(log_alpha, T)
            loss_pm += rnet.prior_matching(nu_z)

            task_preds = alpha.argmax(dim=1, keepdim=False)
            acc = (task_preds == T).float().mean() * 100.

            if i % 10 == 0:
                writer.add_scalar('train_ae/task_%d/accuracy_%d' % (task_id, t),
                                  acc, i)
                if t < task_id:
                    # Add a histogram for the first dimension of replayed
                    # samples. The user can then judge, whether they
                    # originate from the correct input domain.
                    writer.add_histogram('train_ae/task_%d/replay_%d' %
                                         (task_id, t), X[:, 0], i)

        loss = 1. / (task_id+1) * (loss_rec + config.ae_beta_ce * loss_ce +
                                              config.ae_beta_pm * loss_pm)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('train_ae/task_%d/loss' % task_id, loss, i)
            writer.add_scalar('train_ae/task_%d/reconstruction_loss' %
                              task_id, loss_rec, i)
            writer.add_scalar('train_ae/task_%d/cross_entropy' % task_id,
                              loss_ce, i)
            writer.add_scalar('train_ae/task_%d/prior_matching' % task_id,
                              loss_pm, i)

    print('Training recognition network ... Done')

def run():
    """Run the script

    Returns:
        final_mse: Final MSE for each task.
        immediate_mse: MSE achieved directly after training on each task.
        (final_rnet_mse): Final MSE for each task, when using recognition
            model to infer task identity during testing.
    """
    config = train_utils.parse_cmd_arguments(mode='train_regression')

    device, writer = train_utils._setup_environment(config)

    ### Create tasks.
    dhandlers, num_tasks = train_utils._generate_tasks(config)

    ### Generate networks.
    mnet, hnet, rnet = train_utils._generate_networks(config, dhandlers,
        device, create_rnet=config.use_task_detection)

    ### Train on tasks sequentially.
    immediate_mse = np.ones(num_tasks) * -1.
    immediate_weights = np.ones(num_tasks) * -1.
    current_rnet_mse = None

    # DELETEME
    # These baselines correspond to: --no_cuda --main_act=sigmoid
    # --hnet_act=sigmoid --hnet_arch=10,10 --main_arch=10,10 --n_iter=4001
    # --lr_hyper=1e-2 --data_random_seed=42 --beta=0.005 --emb_size=2
    baselines = {
        'Multi-task: Main': [0.000824691285379, 0.00112110003829,
                             0.001513198047178],
        'Immediate MSE - No reg': [0.000851554065594, 0.000517978944117,
                                   0.001182764337864],
        #'Current MSE - No reg': [4.52972555160522, 5.12006094455719,
        #                         0.001182764337864], # Fine-tuning
        'From scratch': [0.000851554065594,0.00098702138057,0.001293907815125]
    }
    baselines = dict()

    for i in range(num_tasks):
        print('### Training on task %d ###' % (i+1))
        data = dhandlers[i]
        # Train the network.
        if config.use_proximal_alg:
            train_proximal(i, data, mnet, hnet, device, config, writer)
        else:
            train_reg(i, data, mnet, hnet, device, config, writer)

        # Train recognition network. Note, our recognition network is completely
        # independent of the hypernetwork or the task embeddings. Hence, to keep
        # the code clean, we can train it separately.
        if config.use_task_detection:
            print('# Training recognition model for task %d' % (i+1))
            train_rnet(i, data, rnet, device, config, writer)

            ### Test networks with recognition network..
            current_rnet_mse, _, _ = test(dhandlers[:(i + 1)], mnet, hnet,
                device, config, writer, rnet=rnet, save_fig=False)
            baselines['Current MSE: Task Recognition'] = current_rnet_mse

            if i == num_tasks-1:
                print('Final MSE values after training on all tasks using ' +
                      'task recognition %s' % str(current_rnet_mse))

        ### Test networks.
        current_mse, immediate_mse, immediate_weights = test(dhandlers[:(i+1)],
            mnet, hnet, device, config, writer, rnet=None,
            immediate_mse=immediate_mse, immediate_weights=immediate_weights,
            baseline_mse = baselines
            #baseline_mse=None if i != num_tasks-1 else baselines)
            )

        if config.train_from_scratch:
            mnet, hnet, rnet = train_utils._generate_networks(config, dhandlers,
                device, create_rnet=config.use_task_detection)

    print('Immediate MSE values after training each task: %s' % \
          np.array2string(immediate_mse, precision=5, separator=','))
    print('Final MSE values after training on all tasks: %s' % \
          np.array2string(current_mse, precision=5, separator=','))
    print('Final MSE mean %.4f (std %.4f).' % (current_mse.mean(),
                                               current_mse.std()))

    writer.close()

    print('Program finished successfully.')

    return current_mse, immediate_mse, current_rnet_mse

if __name__ == '__main__':
    _, _, _ = run()

