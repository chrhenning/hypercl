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
import numpy as np
import torch
from torch.nn import functional as F

from mnets.mnet_interface import MainNetInterface
from toy_example.main_model import MainNetwork
from toy_example.hyper_model import HyperNetwork

def compute_fisher(task_id, data, params, device, mnet, hnet=None,
                   empirical_fisher=True, online=False, gamma=1., n_max=-1,
                   regression=False, time_series=False,
                   allowed_outputs=None, custom_forward=None, custom_nll=None):
    r"""Compute estimates of the diagonal elements of the Fisher information
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
            whole training set (except ``n_max`` is specified).
        params: A list of parameter tensors from the module of which we aim to
            compute the Fisher for. If ``hnet`` is given, then these are assumed
            to be the "theta" parameters, that we pass to the forward function
            of the hypernetwork. Otherwise, these are the "weights" passed to
            the forward method of the main network.
            Note, they might not be detached from their original parameters,
            because we use ``backward()`` on the computational graph to read out
            the ``.grad`` variable.
            Note, the order in which these parameters are passed to this method
            and the corresponding EWC loss function must not change, because
            the index within the "params" list will be used as unique
            identifier.
        device: Current PyTorch device.
        mnet: The main network. If ``hnet`` is ``None``, then ``params`` are
            assumed to belong to this network. The fisher estimate will be
            computed accordingly.
            Note, ``params`` might be the output of the hypernetwork, i.e.,
            weights for a specific task. In this case, "online"-EWC doesn't make
            much sense, as we don't follow the Bayesian view of using the old
            task weights as prior for the current once. Instead, we have a new
            set of weights for all tasks.
        hnet (optional): If given, ``params`` is assumed to correspond to the
            weights "theta" (which does not include task embeddings) of the
            hypernetwork. In this case, the diagonal Fisher entries belong to
            weights of the hypernetwork. The Fisher will then be computed based
            on the probability :math:`p(y \mid x, \text{task\_id})`, where
            ``task_id`` is just a constant input (actually the corresponding
            task embedding) in addition to the training samples :math:`x`.
        empirical_fisher: If ``True``, we compute the fisher based on training
            targets. Note, this has to be ``True`` if ``regression`` is set,
            otherwise the squared norm between main network output and
            "most likely" output is always zero, as they are identical.
        online: If ``True``, then we use online EWC, hence, there is only one
            diagonal Fisher approximation and one target parameter value stored
            at the time, rather than for all previous tasks.
        gamma: The gamma parameter for online EWC, controlling the gradual decay
            of previous tasks.
        n_max (optional): If not ``-1``, this will be the maximum amount of
            training samples considered for estimating the Fisher.
        regression: Whether the task at hand is a classification or regression
            task. If ``True``, a regression task is assumed. For simplicity, we
            assume the following probabilistic model
            :math:`p(y \mid x) = \mathcal{N}\big(f(x), I\big)` with :math:`I`
            being the identity matrix. In this case, the only terms of the log
            probability that influence the gradient are:
            :math:`\log p(y \mid x) = \lVert f(x) - y \rVert^2`
        time_series (bool): If ``True``, the output of the main network
            ``mnet`` is expected to be a time series. In particular, we
            assume that the output is a tensor of shape ``[S, N, F]``,
            where ``S`` is the length of the time series, ``N`` is the batch
            size and ``F`` is the size of each feature vector (e.g., in
            classification, ``F`` would be the number of classes).

            Let :math:`\mathbf{y} = (\mathbf{y}_1, \dots \mathbf{y}_S)` be the
            output of the main network. We denote the parameters ``params`` by
            :math:`\theta` and the input by :math:`\mathbf{x}` (which we do not
            consider as random). We use the following decomposition of the
            likelihood
            
            .. math::
                
                p(\mathbf{y} \mid \theta; \mathbf{x}) =
                \prod_{i=1}^S p(\mathbf{y}_i \mid \mathbf{y}_1, \dots,
                \mathbf{y}_{i-1}, \theta; \mathbf{x}_i)

            **Classification:** If
            :math:`f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta)` denotes the output
            of the main network ``mnet`` for timestep :math:`i` (assuming
            :math:`\mathbf{h}_{i-1}` is the most recent hidden state), we assume

            .. math::

                p(\mathbf{y}_i \mid \mathbf{y}_1, \dots, \mathbf{y}_{i-1},
                \theta; \mathbf{x}_i) \equiv \text{softmax} \big(
                f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta) \big)

            Hence, we assume that we can write the negative log-likelihood (NLL)
            as follows given a label :math:`t \in [1, \dots, F]^S`:

            .. math::

                \text{NLL} &= - \log p(Y = t \mid \theta; \mathbf{x}) \\
                &= \sum_{i=1}^S - \text{softmax} \big(
                f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta)_{t_i} \big) \\
                &= \sum_{i=1}^S \text{cross\_entropy} \big(
                f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta), t_i \big)

            Thus, we simply sum the cross-entropy losses per time-step to
            estimate the NLL, which we then backpropagate through in order to
            compute the diagonal Fisher elements.
        allowed_outputs (optional): A list of indices, indicating which output
            neurons of the main network should be taken into account when
            computing the log probability. If not specified, all output neurons
            are considered.
        custom_forward (optional): A function handle that can replace the
            default procedure of forwarding samples through the given
            network(s).

            The default forward procedure if ``hnet`` is ``None`` is

            .. code:: python

                Y = mnet.forward(X, weights=params)

            Otherwise, the default forward procedure is

            .. code:: python

                weights = hnet.forward(task_id, theta=params)
                Y = mnet.forward(X, weights=weights)

            The signature of this function should be as follows.
                - ``hnet`` is ``None``: :code:`@fun(mnet, params, X)`
                - ``hnet`` is not ``None``:
                  :code:`@fun(mnet, hnet, task_id, params, X)`

            where :code:`X` denotes the input batch to the main network (usually
            consisting of a single sample).

            Example:
                Imagine a situation where the main network uses context-
                dependent modulation (cmp.
                :class:`utils.context_mod_layer.ContextModLayer`) and the
                parameters of these context-mod layers are produced by the
                hypernetwork ``hnet``, whereas the remaining weights of the
                main network ``mnet`` are maintained internally and passed as
                argument ``params`` to this method.

                In particular, we look at a main network that is an instance
                of class :class:`mnets.mlp.MLP`. The forward pass through this
                combination of networks should be handled as follows in order
                to compute the correct fisher matrix:

                .. code:: python

                    def custom_forward(mnet, hnet, task_id, params, X):
                        mod_weights = hnet.forward(task_id)
                        weights = {
                            'mod_weights': mod_weights,
                            'internal_weights': params
                        }
                        Y = mnet.forward(X, weights=weights)
                        return Y
        custom_nll (optional): A function handle that can replace the default 
            procedure of computing the negative-log-likelihood (NLL), which is
            required to compute the Fisher.

            The signature of this function should be as follows:
                :code:`@fun(Y, T, data, allowed_outputs, empirical_fisher)`

            where ``Y`` are the outputs of the main network. Note,
            ``allowed_outputs`` have already been applied to ``Y``, if given.
            ``T`` is the target provided by the dataset ``data``, transformed as
            follows:

            .. code:: python

                T = data.output_to_torch_tensor(batch[1], device,
                                                mode='inference')

            The arguments ``data``, ``allowed_outputs`` and ``empirical_fisher``
            are only passed for convinience (e.g., to apply simple sanity checks
            using assertions).

            The output of the function handle should be the NLL for the given
            sample.
    """
    # Note, this function makes some assumptions about how to use either of
    # these networks. Before adding new main or hypernetwork classes to the
    # assertions, please ensure that this network uses the "forward" functions
    # correctly.
    # If your network does not provide the capability to pass its weights to the
    # forward method, it might be cleaner to implement a separate method,
    # similar to:
    #   https://git.io/fjcnL

    # FIXME The `mnet` should be a subclass of the interface
    # `MainNetInterface`. Though, to ensure downwards compatibility, we allow
    # the deprecated class `MainNetwork` as well for now. However, this class
    # doesn't allow us to check for compatibility (e.g., ensuring that network
    # output is linear).
    assert(isinstance(mnet, MainNetInterface) or \
           isinstance(mnet, MainNetwork))
    assert(hnet is None or isinstance(hnet, HyperNetwork))
    if isinstance(mnet, MainNetInterface):
        assert(mnet.has_linear_out)

    # FIXME The above assertions are not necessary with the new network
    # interfaces, that clearly specify how to use the `forward` methods and how
    # to check the output non-linearity. But someone should carefully check the
    # implementation of this method before adapting the assertions.
    # I just wanna point out that we may wanna provide downwards compatibility
    # as follows as long as not all network types are migrated to the new
    # interface.
    #if not hasattr(mnet, 'has_linear_out'):
    #    pass # TODO new interface not yet available for network type
    #else:
    #    # Knowing the type of output non-linearity gives us a clear way of
    #    # computing the loss for classification tasks.
    #    assert(mnet.has_linear_out)

    assert(hnet is None or task_id is not None)
    assert(regression is False or empirical_fisher)
    assert(not online or (gamma >= 0. and gamma <= 1.))
    assert(n_max is -1 or n_max > 0)

    if time_series and regression:
        raise NotImplementedError('Computing the Fisher for a recurrent ' +
                                  'regression task is not yet implemented.')

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
            if custom_forward is None:
                Y = mnet.forward(X, weights=params)
            else:
                Y = custom_forward(mnet, params, X)
        else:
            if custom_forward is None:
                weights = hnet.forward(task_id, theta=params)
                Y = mnet.forward(X, weights=weights)
            else:
                Y = custom_forward(mnet, hnet, task_id, params, X)

        if not time_series:
            assert(len(Y.shape) == 2)
        else:
            assert(len(Y.shape) == 3)

        if allowed_outputs is not None:
            if not time_series:
                Y = Y[:, allowed_outputs]
            else:
                Y = Y[:, :, allowed_outputs]

        ### Compute negative log-likelihood.
        if custom_nll is not None:
            nll = custom_nll(Y, T, data, allowed_outputs, empirical_fisher)

        elif regression:
            # Note, if regression, we don't have to modify the targets.
            # Thus, through "allowed_outputs" Y has been brought into the same
            # shape as T.

            # The term that doesn't vanish in the gradient of the log
            # probability is the squared L2 norm between Y and T.
            nll = 0.5 * (Y - T).pow(2).sum()

        else:
            # Note, we assume the output of the main network is linear, such
            # that we can compute the log probabilities by applying the log-
            # softmax to these outputs.

            assert(data.classification and len(data.out_shape) == 1)
            if allowed_outputs is not None:
                assert(len(allowed_outputs) == data.num_classes)
                assert(Y.shape[2 if time_series else 1] == data.num_classes)

            # Targets might be labels or one-hot encodings.
            if data.is_one_hot:
                assert(data.out_shape[0] == data.num_classes)
                if time_series:
                    assert(len(T.shape) == 3 and T.shape[2] == data.num_classes)
                    T = torch.argmax(T, 2)
                else:
                    # Note, this function processes always one sample at a time
                    # (batchsize=1), so `T` contains a single number.
                    T = torch.argmax(T)

            # Important, distinguish between empiricial and normal fisher!
            if empirical_fisher:
                if not time_series:
                    # For classification, only the loss associated with the
                    # target unit is taken into consideration.
                    nll = F.nll_loss(F.log_softmax(Y, dim=1),
                                     torch.tensor([T]).to(device))
                else:
                    ll = F.log_softmax(Y, dim=2) # log likelihood for all labels
                    # We need to swap dimenstions from [S, N, F] to [S, F, N].
                    # See documentation of method `nll_loss`.
                    ll = ll.permute(0, 2, 1)
                    nll = F.nll_loss(ll, T, reduction='none')
                    # Mean across batch dimension, but sum across time-series
                    # dimension.
                    assert(len(nll.shape) == 2)
                    nll = nll.mean(dim=1).sum()
            else:
                raise NotImplementedError('Only empirical Fisher is ' +
                                          'implemented so far!')

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

    If `online` is False, this method implements the loss proposed in eq. (3) in
    [EWC2017]_, except for the missing hyperparameter `lambda`.
    
    The online EWC implementation follows eq. (8) from [OnEWC2018]_ (note, that
    lambda does not appear in this equation, but it was used in their
    experiments).

    .. [EWC2017] https://arxiv.org/abs/1612.00796
    .. [OnEWC2018] https://arxiv.org/abs/1805.06370

    Args:
        (....): See docstring of method :func:`compute_fisher`.

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
        task_id: ID of task (only used of `online` is False).
        param_id: Identifier of parameter tensor.
        online: Whether the online EWC algorithm is used.

    Returns:
        (tuple): Tuple containing:

        - **weight_buffer_name**
        - **fisher_estimate_buffer_name**
    """
    task_ident = '' if online else '_task_%d' % task_id

    weight_name = 'ewc_prev{}_weights_{}'.format(task_ident, param_id)
    fisher_name = 'ewc_fisher_estimate{}_weights_{}'.format(task_ident,
                                      param_id)
    return weight_name, fisher_name

def context_mod_forward(mod_weights=None):
    """Create a custom forward function for function :func:`compute_fisher`.

    See argument ``custom_forward`` of function :func:`compute_fisher` for more
    details.

    This is a helper method to quickly retrieve a function handle that manages
    the forward pass for a context-modulated main network.

    We assume that the interface of the main network is similar to the one of
    :meth:`mnets.mlp.MLP.forward`.

    Args:
        mod_weights (optional): If provided, it is assumed that
            :func:`compute_fisher` is called with ``hnet`` set to ``None``.
            Hence, the returned function handle will have the given
            context-modulation pattern hard-coded.
            If left unspecified, it is assumed that a ``hnet`` is passed to
            :func:`compute_fisher` and that this ``hnet`` computes only the
            parameters of all context-mod layers.

    Returns:
        A function handle.
    """
    def hnet_forward(mnet, hnet, task_id, params, X):
        mod_weights = hnet.forward(task_id)
        weights = {
            'mod_weights': mod_weights,
            'internal_weights': params
        }
        Y = mnet.forward(X, weights=weights)
        return Y

    def mnet_only_forward(mnet, params, X):
        weights = {
            'mod_weights': mod_weights,
            'internal_weights': params
        }
        Y = mnet.forward(X, weights=weights)
        return Y

    if mod_weights is None:
        return hnet_forward
    else:
        return mnet_only_forward

def cognet_mse_nll(no_fix_unit_amplification=False):
    r"""Create a custom NLL function for function
    :func:`utils.ewc_regularizer.compute_fisher`.

    Here, we consider a set of cognitive tasks as suggested by

        https://www.nature.com/articles/s41593-018-0310-2

    We assume the network loss is computed as described in section *Training
    procedure* on pg. 12 of the paper lined above.

    Thus the network has an output shape of ``[S, N, F]``, where ``S`` is the
    length of a time sequence, ``N`` is the batch size (we can assume ``N`` is 1
    in function :func:`utils.ewc_regularizer.compute_fisher`) and ``F`` is the
    number of output classes. The network is trained using masked MSE loss.

    Note, there are 9 output classes (8 *output ring units* and 1 *fixation
    output unit*), where the last class (the *fixation output unit*) might be
    treated differently.

    The first 10 timesteps (100 ms) are ignored. The fixation period is defined
    by the timesteps that are associated with label 8.

    During the fixation period, the MSE between ring units and targets
    (which will be zero) will be weighted by 1, whereas the MSE between the
    fixation unit and its target (which will be 1) is weighted by 2.

    During the response period, the weighting will change to 5 for ring units
    and 10 for fixation units.

    Similar to function :func:`utils.ewc_regularizer.compute_fisher` (cmp.
    argument ``time_series``), we adopt the following decomposition of the joint

    .. math::

        p(\mathbf{y} \mid \theta; \mathbf{x}) =
        \prod_{i=1}^S p(\mathbf{y}_i \mid \mathbf{y}_1, \dots,
        \mathbf{y}_{i-1}, \theta; \mathbf{x}_i)

    Since the loss is a masked MSE loss, we assume the predictive distribution
    per time step is a Gaussian
    :math:`\mathcal{N}(\mathbf{\mu}, I \mathbf{\sigma}^2)` with diagonal
    covariance matrix.

    Hence, we can write the NLL for the :math:`i`-th output as follows, assuming
    :math:`\textbf{t}_i` is the corresponding 1-hot target:

    .. math::

        \text{NLL}_i &= - \log p(\mathbf{y}_i = \mathbf{t}_i
        \mid \mathbf{y}_1, \dots, \mathbf{y}_{i-1}, \theta; \mathbf{x}_i)\\
        &= \text{const.} + \frac{1}{2} \sum_{j=0}^8 \frac{1}{\sigma_{i,j}^2}
        \big(f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta)_j - t_{i,j} \big)^2\\
        &= \text{const.} + \sum_{j=0}^8 \tau_{i,j}
        \big(f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta)_j - t_{i,j} \big)^2

    where we defined :math:`\tau_{i,j} \equiv \frac{1}{2 \sigma_{i,j}^2}`.
    Based on the time step :math:`i` and the output unit :math:`j`, we can set
    the variance such that :math:`\tau_{i,j}` corresponds to the masking value
    as defined above.

    The complete NLL over all timesteps is then simply:

    .. math::

        \text{NLL} &= \sum_{i=1}^S \text{NLL}_i \\
        &= \text{const.} + \sum_{i=1}^S \sum_{j=0}^8 \tau_{i,j}
        \big(f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta)_j - t_{i,j} \big)^2

    Note, a mask value of zero (:math:`\tau_{i,j} = 0`) corresponds to infinite
    variance.

    Args:
        no_fix_unit_amplification (bool): If ``True``, then the masking for
            the fixation unit is not amplified (by a factor of 2) as described
            above. Instead, fixation and ring units are treated equally.
    """
    def custom_nll(Y, T, data, allowed_outputs, empirical_fisher):
        # We expect targets to be given as 1-hot encodings.
        assert(np.all(np.equal(list(Y.shape), list(T.shape))))

        # Fixation period is defined by timesteps having label 8.
        labels = torch.argmax(T, 2)

        if no_fix_unit_amplification:
            mask = torch.ones(T.shape[0], T.shape[1])
            mask[labels != 8] = 5
            mask[0:10, :] = 0

            # Make sure that `mask` are broadcastable wrt `Y` and `T`.
            mask = mask.view(T.shape[0], T.shape[1], 1)

        else:
            mask = torch.ones_like(T)
            # Make sure that `labels` can be used to index `mask`.
            labels = labels.view(T.shape[0], T.shape[1], 1)
            labels = labels.expand(mask.size())

            mask[labels != 8] = 5
            mask[0:10, :, :] = 0
            
            mask[:, :, 8] = 2 * mask[:, :, 8]

        nll = (mask * (Y - T)**2).sum()

        return nll

    return custom_nll

if __name__ == '__main__':
    pass


