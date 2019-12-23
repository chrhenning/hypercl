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
@title           :toy_example/train_multitask.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :04/30/2019
@version         :1.0
@python_version  :3.6.8

In this training script, we explore different ways of multi-task learning
(which differs from Continual Learning in that data from all tasks is available
at all times).
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from toy_example import train_utils
import toy_example.train as train_cl
from toy_example.task_recognition_model import RecognitionNet

def train_with_embs(data_handlers, mnet, hnet, device, config, writer,
                   mixed_gradient=True):
    """Train our hypernet - main-net combination on all tasks at once (multi-
    task learning).

    Note, there are several ways to do this. There are methods that ignore the
    task identity of samples, e.g.,

        - training the pure main network on a combined dataset (which is the
          typical baseline in the literature)
        - training a combination of main net and hypernet with a single task
          embeddings, such that the hypernetwork is expected to output a set
          of weights that solves all tasks equally good.

    The above baseline methods have arguably less capacity, which is why one
    would expect that they are no fair comparison to our method.

    An alternative strategy is to keep the task identities (i.e., a task
    embedding for each task) and train on a combined dataset. The pitfall
    with this method is, that using a hypernet with minibatches is not
    readily parallelizable. Hence, in an efficient implementation one would
    select a random task_id per mini-batch and only process samples from this
    task-specific dataset.

    The drawback of this method is, that the gradient would always be computed
    based on a single task (no influence of mixed task signals on the gradient).

    To overcome this problem, one would need to loop over task-embeddings to
    compute the weights for different tasks, which results in linearized
    (inefficient) computation.

    This method allows both types of baseline computation, which keep the task
    identity.

    Args:
        (....): See docstring of method :func:`toy_example.train.train_reg`.
        data_handlers: A list of dataset handlers, one for each task.
        mixed_gradient: If False,in each training iteration a random task will
            be selected and the training step is computed based on samples
            from this task only. If True, random samples from all tasks are
            selected, which means that one has to loop over the generation of
            weights for different tasks (computationally inefficient).
    """
    print('Training network ...')

    mnet.train()
    hnet.train()

    n = len(data_handlers)

    # Number of samples per task per mini-batch if "mixed_gradient" activated.
    sample_size = [config.batch_size // n] * n
    sample_size = [s + (1 if i < config.batch_size % n else 0) for i, s in
                   enumerate(sample_size)]

    # Since we have no regularizers, we can optimize all parameters at once.
    optimizer = optim.Adam(hnet.parameters(), lr=config.lr_hyper)

    for i in range(config.n_iter * n):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            for t in range(n):
                train_cl.evaluate(t, data_handlers[t], mnet, hnet, device,
                                     config, writer, i)
            train_cl.test(data_handlers, mnet, hnet, device, config, writer)
            mnet.train()
            hnet.train()

        if i % 100 == 0:
            print('Training iteration: %d.' % i)

        # Choose number of samples for each task:
        if mixed_gradient:
            n_samples = np.random.permutation(sample_size)
        else:
            n_samples = [0] * n
            n_samples[int(np.random.randint(0, n))] = config.batch_size

        ### Train hypernet.
        optimizer.zero_grad()

        loss = 0
        for t, bs in enumerate(n_samples):
            if bs == 0:
                continue
            data = data_handlers[t]

            batch = data.next_train_batch(bs)
            X = data.input_to_torch_tensor(batch[0], device, mode='train')
            T = data.output_to_torch_tensor(batch[1], device, mode='train')

            weights = hnet.forward(t)
            Y = mnet.forward(X, weights)

            if config.multi_head:
                n_y = data.out_shape[0]
                allowed_outputs = list(range(t*n_y, (t+1)*n_y))
                Y = Y[:, allowed_outputs]

            # Task-specific loss.
            loss += F.mse_loss(Y, T)

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('train/mse_loss', loss, i)

    print('Training network ... Done')

def train_main_only(data_handlers, mnet, device, config, writer):
    """Train main network on all tasks at once (multi-task learning). Note, no
    hypernetwork is used.

    Args:
        (....): See docstring of method :func:`toy_example.train.train_reg`.
        data_handlers: A list of dataset handlers, one for each task.
    """
    print('Training network ...')

    mnet.train()

    n = len(data_handlers)

    optimizer = optim.Adam(mnet.parameters(), lr=config.lr_hyper)

    for i in range(config.n_iter * n):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            for t in range(n):
                train_cl.evaluate(t, data_handlers[t], mnet, None, device,
                                     config, writer, i)
            train_cl.test(data_handlers, mnet, None, device, config, writer)
            mnet.train()

        if i % 100 == 0:
            print('Training iteration: %d.' % i)

        # Choose number of samples for each task:
        _, n_samples = np.unique(np.random.randint(0, high=n,
            size=config.batch_size), return_counts=True)

        ### Train main network.
        optimizer.zero_grad()

        X = torch.empty((config.batch_size, *data_handlers[0].in_shape))
        T = torch.empty((config.batch_size, *data_handlers[0].out_shape))
        m = 0
        for t, bs in enumerate(n_samples):
            if bs == 0:
                continue
            data = data_handlers[t]

            batch = data.next_train_batch(bs)
            X[m:m+bs, :] = data.input_to_torch_tensor(batch[0], device,
                                                      mode='train')
            T[m:m+bs, :] = data.output_to_torch_tensor(batch[1], device,
                                                       mode='train')

            m += bs

        Y = mnet.forward(X)

        if config.multi_head:
            n_y = data.out_shape[0]

            Y_full = Y
            Y = torch.empty((Y_full.shape[0], n_y))

            m = 0
            for t, bs in enumerate(n_samples):
                allowed_outputs = list(range(t*n_y, (t+1)*n_y))
                Y[m:m+bs, :] = Y_full[m:m+bs, allowed_outputs]

                m += bs

        # Task-specific loss.
        loss = F.mse_loss(Y, T)

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('train/mse_loss', loss, i)

    print('Training network ... Done')

def train_without_embs(data_handlers, mnet, hnet, device, config, writer):
    """Train our hypernet - main-net combination on all tasks at once (multi-
    task learning), using only one task embedding. Hence, the hypernetwork has
    to learn to produce a set of weights, that solves all tasks at once.


    Args:
        (....): See docstring of method :func:`toy_example.train.train_reg`.
        data_handlers: A list of dataset handlers, one for each task.
    """
    print('Training network ...')

    mnet.train()
    hnet.train()

    n = len(data_handlers)

    # Since we have no regularizers, we can optimize all parameters at once.
    optimizer = optim.Adam(hnet.parameters(), lr=config.lr_hyper)

    for i in range(config.n_iter * n):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            for t in range(n):
                train_cl.evaluate(t, data_handlers[t], mnet, hnet, device,
                                     config, writer, i)
            train_cl.test(data_handlers, mnet, hnet, device, config, writer)
            mnet.train()
            hnet.train()

        if i % 100 == 0:
            print('Training iteration: %d.' % i)

        # Choose number of samples for each task:
        _, n_samples = np.unique(np.random.randint(0, high=n,
            size=config.batch_size), return_counts=True)

        ### Train hypernet.
        optimizer.zero_grad()
        
        X = torch.empty((config.batch_size, *data_handlers[0].in_shape))
        T = torch.empty((config.batch_size, *data_handlers[0].out_shape))
        m = 0
        for t, bs in enumerate(n_samples):
            if bs == 0:
                continue
            data = data_handlers[t]

            batch = data.next_train_batch(bs)
            X[m:m+bs, :] = data.input_to_torch_tensor(batch[0], device,
                                                      mode='train')
            T[m:m+bs, :] = data.output_to_torch_tensor(batch[1], device,
                                                       mode='train')

            m += bs

        weights = hnet.forward(0)
        Y = mnet.forward(X, weights)

        if config.multi_head:
            n_y = data.out_shape[0]

            Y_full = Y
            Y = torch.empty((Y_full.shape[0], n_y))

            m = 0
            for t, bs in enumerate(n_samples):
                allowed_outputs = list(range(t*n_y, (t+1)*n_y))
                Y[m:m+bs, :] = Y_full[m:m+bs, allowed_outputs]

                m += bs

        loss = F.mse_loss(Y, T)

        loss.backward()
        optimizer.step()

        # Ensure that all embeddings are the same (note, that the testing and
        # evaluation methods require an embedding for each task).
        tembs = hnet.get_task_embs()
        for t in range(1, len(tembs)):
            tembs[t].data = tembs[0].data

        if i % 10 == 0:
            writer.add_scalar('train/mse_loss', loss, i)

    print('Training network ... Done')

def train_rnet(data_handlers, mnet, hnet, rnet, device, config, writer):
    """Train the recognition network. However, for multitask learning, there is
    no need to prevent catastophic forgetting in the task recognition network,
    hence, no replay network (decoder) is needed.

    Note, mnet and hnet are only needed for testing purposes.

    Args:
        (....): See docstring of method :func:`toy_example.train.train_reg`.
        data_handlers: A list of dataset handlers, one for each task.
        rnet: The recognition network.
    """
    print('Training recognition network ...')
    
    n = len(data_handlers)

    rnet.train()

    optimizer = optim.Adam(rnet.parameters(), lr=config.lr_ae)

    for i in range(config.n_iter_ae * n):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            test_mse, _, _ = train_cl.test(data_handlers, mnet, hnet, device,
                                           config, writer, rnet=rnet)
            print('Current MSE values using task recognition %s' % 
                  str(test_mse))
            rnet.train()

        if i % 100 == 0:
            print('Training iteration: %d.' % i)

        ### Train theta.
        optimizer.zero_grad()

        loss = 0

        for t in range(n):
            # Task recognition targets:
            T = torch.ones(config.batch_size, dtype=torch.int64).to(device) * t

            data = data_handlers[t]
            batch = data.next_train_batch(config.batch_size)
            X = data.input_to_torch_tensor(batch[0], device, mode='train')

            alpha, h_alpha = rnet.forward(X)

            loss += RecognitionNet.task_cross_entropy(F.log_softmax(h_alpha,
                                                                    dim=1), T)

            task_preds = alpha.argmax(dim=1, keepdim=False)
            acc = (task_preds == T).float().mean() * 100.

            if i % 10 == 0:
                writer.add_scalar('train_ae/accuracy_%d' % (t), acc, i)

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('train_ae/loss', loss, i)

    print('Training recognition network ... Done')

def run():
    """Run the script

    Returns:
        (tuple): Tuple containing:

        - **final_mse**: Final MSE for each task.
        - **final_rnet_mse** (optional): Final MSE for each task, when using
          recognition model to infer task identity during testing.
    """
    config = train_utils.parse_cmd_arguments(mode='train_mt_regression')
    assert(config.method in range(3))

    device, writer = train_utils._setup_environment(config)

    if config.method in [0, 2] and config.use_task_detection:
        raise ValueError('The selected multitask method is not conditioned ' +
                         'on a task. Hence, a recognition model is not ' +
                         'required.')

    ### Create tasks.
    dhandlers, num_tasks = train_utils._generate_tasks(config)

    ### Generate networks.
    # Note, if we use a recognition network, then we use a custom one, as we
    # don't need an autoencoder (no replay needed in multitask learning).
    mnet, hnet, rnet = train_utils._generate_networks(config, dhandlers, device,
        create_hnet=config.method != 0, create_rnet=config.use_task_detection,
        no_replay=True)

    ### Train on tasks in parallel.
    current_rnet_mse = None

    if config.method == 0:
        train_main_only(dhandlers, mnet, device, config, writer)
    elif config.method == 1:
        train_with_embs(dhandlers, mnet, hnet, device, config, writer)

        if config.use_task_detection:
            train_rnet(dhandlers, mnet, hnet, rnet, device, config, writer)

            current_rnet_mse, _, _ = train_cl.test(dhandlers, mnet, hnet,
                device, config, writer, rnet=rnet, save_fig=False)
            print('Final MSE values after training on all tasks using ' +
                  'task recognition %s' % str(current_rnet_mse))

    elif config.method == 2:
        train_without_embs(dhandlers, mnet, hnet, device, config, writer)

    ### Test networks.
    current_mse, _, _ = train_cl.test(dhandlers, mnet, hnet, device, config,
                                       writer)

    print('Final MSE values after training on all tasks: %s' % \
          np.array2string(current_mse, precision=5, separator=','))
    print('Final MSE mean %.4f (std %.4f).' % (current_mse.mean(),
                                               current_mse.std()))

    writer.close()

    print('Program finished successfully.')

    return current_mse, current_rnet_mse

if __name__ == '__main__':
    _, _ = run()

