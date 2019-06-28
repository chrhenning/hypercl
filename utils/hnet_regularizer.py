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
"""
@title           :utils/hnet_regularizer.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :06/05/2019
@version         :1.0
@python_version  :3.6.8

We summarize our own regularizers in this module. These regularizer ensure that
the output of a hypernetwork don't change.
"""

import torch
import numpy as np

from utils.module_wrappers import CLHyperNetInterface

def get_current_targets(task_id, hnet):
    """For all j < task_id, compute the output of the hypernetwork. This
    output will be detached from the graph and cloned before being added to the
    return list of this function.

    Note, if these targets don't change during training, it would be more memory
    efficient to store the weights theta* of the hypernetwork (which is a fixed
    amount compared to the variable number of tasks). Though, it is more
    computationally expensive to recompute h(c_j, theta*) for all j < task_id
    everytime the target is needed.

    Note, this function sets the hypernet temporarily in eval mode.

    Args:
        task_id: The ID of the current task.
        hnet: An instance of the hypernetwork before learning a new task
            (i.e., the hypernetwork has the weights theta* necessary to
            compute the targets).

    Returns:
        An empty list, if task_id == 0. Otherwise, a list of task_id-1 targets.
        These targets can be passed to the method "calc_fix_target_reg" while
        training on the new task.
    """
    # We temporarily switch to eval mode for target computation (e.g., to get
    # rid of training stochasticities such as dropout).
    hnet_mode = hnet.training
    hnet.eval()

    ret = []
    
    for j in range(task_id):
        W = hnet.forward(task_id=j)
        ret.append([d.detach().clone() for d in W])

    hnet.train(mode=hnet_mode)

    return ret

def calc_jac_reguarizer(hnet, task_id, dTheta, device):
    """Compute the CL regularzier, which is a sum over all previous task
    ID's, that enforces that the norm of the matrix product of hypernet
    Jacobian and dTheta is small.

    I.e., for all j < task_id minimize the following norm:
        || J_h(c_j, theta) dTheta ||^2
    where theta (and dTheta) is assumed to be vectorized.

    This regularizer origins in a first-order Taylor approximation of the
    regularizer:
        || h(c_j, theta) - h(c_j, theta + dTheta) ||^2

    Args:
        See parameters of method "calc_fix_target_reg".
        device: Current PyTorch device.

    Returns:
        The value of the regularizer.
    """
    assert(task_id > 0)
    assert(hnet.has_theta) # We need parameters to be regularized.

    reg = 0

    for i in range(task_id): # For all previous tasks.
        W = torch.cat([w.view(-1) for w in hnet.forward(task_id=i)])

        # Problem, autograd.grad() accumulates all gradients with
        # respect to each w in "W". Hence, we don't get a Jacobian but a
        # tensor of the same size as "t", where the partials of all W's
        # with respect to this "t" are summed.
        #J = torch.autograd.grad(W, t,
        #    grad_outputs=torch.ones(W.size()).to(device),
        #    retain_graph=True, create_graph=True, only_inputs=True)[0]
        #
        # Hence, to get the actual jacobian, we have to iterate over all
        # individual outputs of the hypernet.
        for wind, w in enumerate(W):
            tmp = 0
            for tind, t in enumerate(hnet.theta):
                partial = torch.autograd.grad(w, t, grad_outputs=None,
                    retain_graph=True, create_graph=True,
                    only_inputs=True)[0]

                # Intuitively, "partial" represents part of a row in the
                # Jacobian (if dTheta would have been linearized to a
                # vector). To compute the matrix vector product (Jacobian
                # times dTheta), we have to sum over the element-wise
                # product of rows from the Jacobian with dTheta.
                tmp += torch.mul(partial, dTheta[tind]).sum()

            # Since we are interested in computing the squared L2 norm of
            # the matrix vector product: Jacobian times dTheta,
            # we have to simply sum the sqaured dot products between all
            # rows of the Jacobian with dTheta.
            reg += torch.pow(tmp, 2)

    # Normalize by the number of tasks.
    return reg / task_id

def calc_value_preserving_reg(hnet, task_id, dTheta):
    """This regularizer simply restricts a change in output-mapping for
    previous task embeddings. I.e., for all j < task_id minimize:
        || h(c_j, theta) - h(c_j, theta + dTheta) ||^2

    Args:
        See parameters of method "calc_fix_target_reg".

    Returns:
        The value of the regularizer.
    """
    assert(task_id > 0)
    assert(hnet.has_theta) # We need parameters to be regularized.

    reg = 0

    for i in range(task_id): # For all previous tasks.
        W_prev = torch.cat([w.view(-1) for w in hnet.forward(task_id=i)])
        W_new = torch.cat([w.view(-1) for w in hnet.forward(task_id=i,
                                                            dTheta=dTheta)])

        reg += (W_prev - W_new).pow(2).sum()

    return reg / task_id

def calc_fix_target_reg(hnet, task_id, targets=None, dTheta=None, dTembs=None,
                        mnet=None, inds_of_out_heads=None,
                        fisher_estimates=None, prev_theta=None,
                        prev_task_embs=None, batch_size=None):
    """This regularizer simply restricts the output-mapping for
    previous task embeddings. I.e., for all j < task_id minimize:
        || target_j - h(c_j, theta + dTheta) ||^2
    where c_j is the current task embedding for task j (and we assumed that
    "dTheta" was passed).

    Args:
        hnet: The hypernetwork whose output should be regularized. Has to
            implement the interface CLHyperNetInterface.
        task_id: The ID of the current task (the one that is used to
            compute dTheta.
        targets: A list of outputs of the hypernetwork. Each list entry must
            have the output shape as returned by the forward method of this
            class. Note, this method doesn't detach targets. If desired,
            that should be done before calling this method.
        dTheta (optional): The current direction of weight change for the
            internal weights of the hypernetwork evaluated on the task-specific
            loss, i.e., the weight change that would be applied to theta. This
            regularizer aims to modify this direction, such that the hypernet
            output for embeddings of previous tasks remains unaffected.
            Note, this function does not detach dTheta. It is up to the
            user to decide whether dTheta should be a constant vector or
            might depend on parameters of the hypernet.
        dTembs (optional): The current direction of weight change for the task
            embeddings of all tasks been learned already.
            See dTheta for details.
        mnet: Instance of the main network. Has to be given if "allowed_outputs"
            are specified.
        inds_of_out_heads: (optional): List of lists of integers, denoting which
            output neurons of the fully-connected output layer of the main
            network belong to the output head of all already learned task ids.
            This will ensure that only weights of output neurons involved in
            solving a task are regularized.
            Note, this may only be used for main networks that have a fully-
            connected output layer.
        fisher_estimates (optional): A list of list of tensors, containing
            estimates of the Fisher Information matrix for each weight
            tensor in the main network and each task.
                len(fisher_estimates) == task_id
            The Fisher estimates are used as importance weights for single
            weights when computing the regularizer.
        prev_theta (optional): If given, "prev_task_embs" but not "targets"
            has to be specified. "prev_theta" is expected to be the internal
            weights theta prior to learning the current task. Hence, it can be
            used to compute the targets on the fly (which is more memory
            efficient (constant memory), but more computationally demanding).
            The computed targets will be detached from the computational graph.
            Independent of the current hypernet mode, the targets are computed
            in "eval" mode.
        prev_task_embs (optional): If given, "prev_theta" but not "targets"
            has to be specified. "prev_task_embs" are the task embeddings 
            learned prior to learning the current task. It is sufficient to
            only pass the task embeddings for tasks with ID smaller than the
            current one (only those tasks that are regularized).
            See docstring of "prev_theta" for more details.
        batch_size (optional): If specified, only a random subset of previous
            task mappings is regularized. If the given number is bigger than the
            number of previous tasks, all previous tasks are regularized.

    Returns:
        The value of the regularizer.
    """
    assert(isinstance(hnet, CLHyperNetInterface))
    assert(task_id > 0)
    assert(hnet.has_theta) # We need parameters to be regularized.
    assert(len(targets) == task_id)
    assert(inds_of_out_heads is None or mnet is not None)
    assert(inds_of_out_heads is None or len(inds_of_out_heads) >= task_id)
    assert(targets is None or (prev_theta is None and prev_task_embs is None))
    assert(prev_theta is None or prev_task_embs is not None)
    assert(prev_task_embs is None or len(prev_task_embs) >= task_id)
    assert(dTembs is None or len(dTembs) >= task_id)

    # Number of tasks to be regularized.
    num_regs = task_id
    ids_to_reg = list(range(num_regs))
    if batch_size is not None:
        if num_regs > batch_size:
            ids_to_reg = np.random.choice(num_regs, size=batch_size,
                                          replace=False).tolist()
            num_regs = batch_size

    reg = 0

    for i in ids_to_reg:
        if dTembs is None:
            weights_predicted = hnet.forward(task_id=i, dTheta=dTheta)
        else:
            temb = hnet.get_task_emb(i) + dTembs[i]
            weights_predicted = hnet.forward(dTheta=dTheta, task_emb=temb)

        if targets is not None:
            target = targets[i]
        else:
            # Compute targets in eval mode!
            hnet_mode = hnet.training
            hnet.eval()

            # Compute target on the fly using previous hnet.
            target = hnet.forward(theta=prev_theta, task_emb=prev_task_embs[i])
            target = [d.detach().clone() for d in target]

            hnet.train(mode=hnet_mode)

        if inds_of_out_heads is not None:
            # Regularize all weights of the main network except for the weights
            # belonging to output heads of the target network other than the
            # current one (defined by task id).
            W_target = flatten_and_remove_out_heads(mnet, target,
                                                    inds_of_out_heads[i])
            W_predicted = flatten_and_remove_out_heads(mnet, weights_predicted,
                                                       inds_of_out_heads[i])
        else:
            # Regularize all weights of the main network.
            W_target = torch.cat([w.view(-1) for w in target])
            W_predicted = torch.cat([w.view(-1) for w in weights_predicted])

        if fisher_estimates is not None:
            _assert_shape_equality(weights_predicted, fisher_estimates[i])
            FI = torch.cat([w.view(-1) for w in fisher_estimates[i]])

            reg += (FI * (W_target - W_predicted).pow(2)).sum()
        else:
            reg += (W_target - W_predicted).pow(2).sum()

    return reg / num_regs

def _assert_shape_equality(list1, list2):
    """Ensure that 2 lists of tensors have the same shape."""
    assert(len(list1) == len(list2))
    for i in range(len(list1)):
        assert(np.all(np.equal(list(list1[i].shape), list(list2[i].shape))))

def flatten_and_remove_out_heads(mnet, weights, allowed_outputs):
    """Flatten a list of target network tensors to a single vector, such that
    output neurons that belong to other than the current output head are
    dropped.

    Note, this method assumes that the main network has a fully-connected output
    layer.

    Args:
        mnet: Main network instance.
        weights: A list of weight tensors of the main network (must adhere the
            corresponding weight shapes).
        allowed_outputs: List of integers, denoting which output neurons of
            the fully-connected output layer belong to the current head.

    Returns:
        The flattened weights with those output weights not belonging to the
        current head being removed.
    """
    assert(mnet.has_fc_out)

    obias_ind = len(weights)-1 if mnet.has_bias else -1
    oweights_ind = len(weights)-2 if mnet.has_bias else len(weights)-1

    ret = []
    for i, w in enumerate(weights):
        if i == obias_ind: # Output bias
            ret.append(w[allowed_outputs])
        elif i == oweights_ind: # Output weights
            ret.append(w[allowed_outputs, :].view(-1))
        else:
            ret.append(w.view(-1))

    return torch.cat(ret)

if __name__ == '__main__':
    pass


