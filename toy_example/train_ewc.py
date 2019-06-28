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
@title           :toy_example/train_ewc.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :05/08/2019
@version         :1.0
@python_version  :3.6.8

In this script, we train the main network only using the Elastic Weight
Consolidation (EWC) algorithm.

Note, if using regression tasks with overlapping input domain, you should use
a multi-head setup, otherwise the output can't be meaningful!
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from toy_example import train_utils
import toy_example.train as train_cl
import utils.ewc_regularizer as ewc

def train_ewc(task_id, data, mnet, device, config, writer):
    """Train the main network in a continual learning setup using the EWC
    regularizer to prevent catastrophic forgetting.

    loss = task_loss + beta * ewc_regularizer.

    Args:
        See docstring of method train_reg.
    """
    print('Training network ...')

    mnet.train()

    allowed_outputs = None
    if config.multi_head:
        n_y = data.out_shape[0]
        allowed_outputs = list(range(task_id*n_y, (task_id+1)*n_y))

    optimizer = optim.Adam(mnet.parameters(), lr=config.lr_hyper)

    for i in range(config.n_iter):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            train_cl.evaluate(task_id, data, mnet, None, device, config, writer,
                              i)
            mnet.train()

        if i % 100 == 0:
            print('Training iteration: %d.' % i)

        ### Train theta.
        optimizer.zero_grad()

        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')

        Y = mnet.forward(X)
        if config.multi_head:
            Y = Y[:, allowed_outputs]

        # Task-specific loss.
        loss_task = F.mse_loss(Y, T)

        loss_reg = 0

        if task_id > 0 and config.beta > 0:
            loss_reg = ewc.ewc_regularizer(task_id, mnet.weights, mnet,
                online=config.online_ewc, gamma=config.gamma)

        loss = loss_task + config.beta * loss_reg
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('train/task_%d/mse_loss' % task_id, loss_task, i)
            writer.add_scalar('train/task_%d/regularizer' % task_id, loss_reg,
                              i)
            writer.add_scalar('train/task_%d/full_loss' % task_id, loss, i)

    ## Estimate diagonal Fisher elements.
    ewc.compute_fisher(task_id, data, mnet.weights, device, mnet,
        empirical_fisher=True, online=config.online_ewc, gamma=config.gamma,
        n_max=config.n_fisher, regression=True, allowed_outputs=allowed_outputs)

    print('Training network ... Done')

def run():
    """Run the script

    Returns:
        final_mse: Final MSE for each task.
        immediate_mse: MSE achieved directly after training on each task.
    """
    config = train_utils.parse_cmd_arguments(mode='train_ewc_regression')

    device, writer = train_utils._setup_environment(config)

    ### Create tasks.
    dhandlers, num_tasks = train_utils._generate_tasks(config)

    ### Generate networks.
    mnet, _, _ = train_utils._generate_networks(config, dhandlers,
        device, create_hnet=False, create_rnet=False)

    ### Train on tasks sequentially.
    immediate_mse = np.ones(num_tasks) * -1.

    for i in range(num_tasks):
        print('### Training on task %d ###' % (i+1))
        data = dhandlers[i]
        # Train the network.
        train_ewc(i, data, mnet, device, config, writer)

        ### Test networks.
        current_mse, immediate_mse, _ = train_cl.test( \
            dhandlers[:(i+1)], mnet, None, device, config, writer, rnet=None,
            immediate_mse=immediate_mse)

        if config.train_from_scratch:
            mnet, _, _ = train_utils._generate_networks(config, dhandlers,
                device, create_hnet=False, create_rnet=False)

    print('Immediate MSE values after training each task: %s' % \
          np.array2string(immediate_mse, precision=5, separator=','))
    print('Final MSE values after training on all tasks: %s' % \
          np.array2string(current_mse, precision=5, separator=','))
    print('Final MSE mean %.4f (std %.4f).' % (current_mse.mean(),
                                               current_mse.std()))

    writer.close()

    print('Program finished successfully.')

    return current_mse, immediate_mse

if __name__ == '__main__':
    _, _ = run()



