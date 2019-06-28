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
@title           :utils/ewc_regularizer.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :05/07/2019
@version         :1.0
@python_version  :3.6.8

Implementation of EWC:
    https://arxiv.org/abs/1612.00796

Note, these implementation are based on the descriptions provided in:
    https://arxiv.org/abs/1809.10635

The code is inspired by the corresponding implementation:
    https://git.io/fjcnL
"""
import torch
import torch.nn.functional as F

from toy_example.main_model import MainNetwork
from toy_example.hyper_model import HyperNetwork

def compute_fisher(task_id, data, params, device, mnet, hnet=None,
                   empirical_fisher=True, online=False, gamma=1., n_max=-1,
                   regression=False, allowed_outputs=None):
    """Compute estimates of the diagonal elements of the Fisher information
    matrix, as needed as importance-weights by elastic weight consolidation
    (EWC).

    Note, this method registers buffers in the given module (storing the
    current parameters and the estimate of the Fisher diagonal elements), i.e.,
    the "mnet" if "hnet" is None, otherwise the "hnet".

    Args:
        task_id: The ID of the current task, needed to store the computed
            tensors with a unique name. When "hnet" is given, it is used as
            input to the "hnet" forward method to select the current task
            embedding.
        data: A data handler. We will compute the Fisher estimate across the
            whole training set (except "n_max" is specified).
        params: A list of parameter tensors from the module of which we aim to
            compute the Fisher for. If "hnet" is given, then these are assumed
            to be the "theta" parameters, that we pass to the forward function
            of the hypernetwork. Otherwise, these are the "weights" passed to
            the forward function of the main network.
            Note, they might not be detached from their original parameters,
            because we use backward() on the computational graph to read out
            the ".grad" variable.
            Note, the order in which these parameters are passed to this method
            and the corresponding EWC loss function must not change, because
            the index within the "params" list will be used as unique
            identifier.
        device: Current PyTorch device.
        mnet: The main network. If hnet is None, then "params" are assumed to
            belong to this network. The fisher estimate will be computed
            accordingly.
            Note, "params" might be the output of the hypernetwork, i.e.,
            weights for a specific task. In this case, "online"-EWC doesn't make
            much sense, as we don't follow the Bayesian view of using the old
            task weights as prior for the current once. Instead, we have a new
            set of weights for all tasks.
        hnet (optional): If given, "params" is assumed to correspond to the
            weights theta (which does not include task embeddings) of the
            hypernetwork. In this case, the diagonal Fisher entries belong to
            weights of the hypernetwork. The Fisher will then be computed based
            on the probability p(y | x, task_id), where "task_id" is just a
            constant input (actually the corresponding task embedding) in
            addition to the training samples x.
        empirical_fisher: If True, we compute the fisher based on training
            targets. Note, this has to be True if "regression" is set, otherwise
            the squared norm between main network output and "most likely"
            output is always zero, as they are identical.
        online: If True, then we use online EWC, hence, there is only one
            diagonal Fisher approximation and one target parameter value stored
            at the time, rather than for all previous tasks.
        gamma: The gamma parameter for online EWC, controlling the gradual decay
            of previous tasks.
        n_max (optional): If not "-1", this will be the maximum amount of
            training samples considered for estimating the Fisher.
        regression: Whether the task at hand is a classification or regression
            task. If True, a regression task is assumed. For simplicity, we
            assume the following probabilistic model p(y | x) = N(f(x), I) with
            I be the identity matrix. In this case, the only terms of the log
            probability that influence the gradient are:
                log p(y | x) = || f(x) - y ||^2
        allowed_outputs (optional): A list of indices, indicating which output
            neurons of the main network should be taken into account when
            computing the log probability. If not specified, all output neurons
            are considered.
    """
    # Note, this function makes some assumptions about how to use either of
    # these networks. Before adding new main or hypernetwork classes to the
    # assertions, please ensure that this network uses the "forward" functions
    # correctly.
    # If your network does not provide the capability to pass its weights to the
    # forward method, it might be cleaner to implement a separate method,
    # similar to:
    #   https://git.io/fjcnL
    assert(isinstance(mnet, MainNetwork))
    assert(hnet is None or isinstance(hnet, HyperNetwork))

    assert(hnet is None or task_id is not None)
    assert(regression is False or empirical_fisher)
    assert(not online or (gamma >= 0. and gamma <= 1.))
    assert(n_max is -1 or n_max > 0)

    n_samples = data.num_train_samples
    if n_max != -1:
        n_samples = min(n_samples, n_max)

    mnet_mode = mnet.training
    mnet.eval()
    if hnet is not None:
        hnet_mode = hnet.training
        hnet.eval()

    fisher = []
    for p in params:
        fisher.append(torch.zeros_like(p))

        assert(p.requires_grad) # Otherwise, we can't compute the Fisher.

    # Ensure, that we go through all training samples (note, that training
    # samples are always randomly shuffled when using "next_train_batch", but
    # we always go though the complete batch before reshuffling the samples.)
    # If n_max was specified, we always go through a different random subsample
    # of the training set.
    data.reset_batch_generator(train=True, test=False, val=False)

    # Since the PyTorch grad function accumulates gradients, we have to go
    # through single training samples.
    for s in range(n_samples):
        batch = data.next_train_batch(1)
        X = data.input_to_torch_tensor(batch[0], device, mode='inference')
        T = data.output_to_torch_tensor(batch[1], device, mode='inference')

        if hnet is None:
            Y = mnet.forward(X, weights=params)
        else:
            weights = hnet.forward(task_id, theta=params)
            Y = mnet.forward(X, weights=weights)

        if allowed_outputs is not None:
            Y = Y[:, allowed_outputs]

        ### Compute negative log-likelihood.
        if regression:
            # Note, if regression, we don't have to modify the targets.
            # Thus, through "allowed_outputs" Y has been brought into the same
            # shape as T.

            # The term that doesn't vanish in the gradient of the log
            # probability is the squared L2 norm between Y and T.
            nll = (Y - T).pow(2).sum()

        else:
            # TODO Implement
            # Note, when computing the negative log likelihood, be careful what
            # the output of the main network is (softmax, log-softmax, logit,
            # ...).
            raise NotImplementedError('Method not implemented for ' +
                                      'classification networks.')

            # Note, targets might be labels or one-hot encodings.
            if allowed_outputs is not None:
                pass

            # TODO distinguish between empiricial and normal fisher!
            if empirical_fisher:
                pass
            else:
                pass

            nll = None

        ### Compute gradient of negative log likelihood to estimate Fisher
        mnet.zero_grad()
        if hnet is not None:
            hnet.zero_grad()
        torch.autograd.backward(nll, retain_graph=False, create_graph=False)

        for i, p in enumerate(params):
            fisher[i] += torch.pow(p.grad.detach(), 2)

        # This version would not require use to call zero_grad and hence, we
        # wouldn't fiddle with internal variables, but it would require us to
        # loop over tensors and retain the graph in between.
        #for p in params:
        #    g = torch.autograd.grad(nll, p, grad_outputs=None,
        #                retain_graph=True, create_graph=False,
        #                only_inputs=True)[0]
        #    fisher[i] += torch.pow(g.detach(), 2)

    for i in range(len(params)):
        fisher[i] /= n_samples

    ### Register buffers to store current task weights as well as the Fisher.
    net = mnet
    if hnet is not None:
        net = hnet
    for i, p in enumerate(params):
        buff_w_name, buff_f_name = _ewc_buffer_names(task_id, i, online)
        
        # We use registered buffers rather than class members to ensure that
        # these variables appear in the state_dict and are thus written into
        # checkpoints.
        net.register_buffer(buff_w_name, p.detach().clone())

        # In the "online" case, the old fisher estimate buffer will be
        # overwritten.
        if online and task_id > 0:
            prev_fisher_est = getattr(net, buff_f_name)

            # Decay of previous fisher.
            fisher[i] += gamma * prev_fisher_est

        net.register_buffer(buff_f_name, fisher[i].detach().clone())

    mnet.train(mode=mnet_mode)
    if hnet is not None:
        hnet.train(mode=hnet_mode)

def ewc_regularizer(task_id, params, mnet, hnet=None,
                    online=False, gamma=1.):
    """Compute the EWC regularizer, that can be added to the remaining loss.
    Note, the hyperparameter, that trades-off the regularization strength is
    not yet multiplied by the loss.

    This loss assumes an appropriate use of the method "compute_fisher". Note,
    for the current task "compute_fisher" has to be called after calling this
    method.

    If "online" is False, this method implements the loss proposed in eq. (3) of
        https://arxiv.org/abs/1612.00796
    except for the missing hyperparameter "lambda".
    
    The online EWC implementation follows eq. (8) from
        https://arxiv.org/abs/1805.06370
    (note, that lambda does not appear in this equation, but it was used in
    their experiments)

    Args:
        See docstring of method "compute_fisher"

    Returns:
        EWC regularizer.
    """
    assert(task_id > 0)

    net = mnet
    if hnet is not None:
        net = hnet

    ewc_reg = 0

    num_prev_tasks = 1 if online else task_id
    for t in range(num_prev_tasks):
        for i, p in enumerate(params):
            buff_w_name, buff_f_name = _ewc_buffer_names(t, i, online)

            prev_weights = getattr(net, buff_w_name)
            fisher_est = getattr(net, buff_f_name)
            # Note, since we haven't called "compute_fisher" yet, the forgetting
            # scalar has been multiplied yet.
            if online:
                fisher_est *= gamma

            ewc_reg += (fisher_est * (p - prev_weights).pow(2)).sum()

    # Note, the loss proposed in the original paper is not normalized by the
    # number of tasks
    #return ewc_reg / num_prev_tasks / 2.
    return ewc_reg / 2.

def _ewc_buffer_names(task_id, param_id, online):
    """The names of the buffers used to store EWC variables.

    Args:
        task_id: ID of task (only used of "online" is False).
        param_id: Identifier of parameter tensor.
        online: Whether the online EWC algorithm is used.

    Returns:
        weight_buffer_name, fisher_estimate_buffer_name
    """
    task_ident = '' if online else '_task_%d' % task_id

    weight_name = 'ewc_prev{}_weights_{}'.format(task_ident, param_id)
    fisher_name = 'ewc_fisher_estimate{}_weights_{}'.format(task_ident,
                                      param_id)
    return weight_name, fisher_name

if __name__ == '__main__':
    pass


