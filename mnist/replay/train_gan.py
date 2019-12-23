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
# @created         :07/10/2019
# @version         :1.0
# @python_version  :3.6.8

"""
Continual learning of MNIST GAN with hypernetworks
---------------------------------------------------

An implementation of a simple fully-connected MNIST GAN. The goal of this
script is to provide a sanity check, that an MNIST GAN can be realized through
a hypernetwork, i.e., a hypernetwork that produces the weights of the generator.
"""

# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import torch
import torch.optim as optim
import os
import numpy as np

from utils import gan_helpers 
from mnist.plotting import _viz_training, _plotImages

import utils.hnet_regularizer as hreg
import utils.optim_step as opstep

def test(dis, gen, g_hnet, device, config, writer, train_iter=None, 
                                                                condition=None):
    """ Test the MNIST GAN - here we only sample from a fixed noise to compare
    images qualitatively. One should also keep track of the GAN loss 
    error of e.g. a test set.

    Args:
        (....): See docstring of function 
            :func:`mnist.replay.train_replay.train`.
        train_iter: The current training iteration.
        condition: Condition (class/task) we are currently training.
    """

    if train_iter is None:
        print('### Final test run ...')
        train_iter = config.n_iter
    else:
        print('# Testing network before running training step %d ...' % \
              train_iter)
    # if no condition is given, we iterate over all (trained) embeddings
    if condition is None:
        condition = config.num_embeddings - 1 

    # eval all nets
    dis.eval()
    gen.eval()
    if g_hnet is not None:
        g_hnet.eval()

    with torch.no_grad():
        # iterate over all conditions
        for m in range(condition + 1):
            # Get pre training saved noise
            z = config.test_z[m]
            X_fake = sample(gen, g_hnet, config, m, device, z = z)
            X_fake = X_fake*2-1
            if config.show_plots:
                fig_real = _plotImages(X_fake, config)
                writer.add_figure('test_cond_' + str(m) + 
                                    '_sampled_after_'+str(condition), fig_real, 
                                                global_step=train_iter)
                if train_iter == config.n_iter:
                    writer.add_figure('test_cond_final_' + str(m) + 
                                    '_sampled_after_'+str(condition), fig_real, 
                                                global_step=train_iter)
            # TODO test GAN loss           

def sample(gen, g_hnet, config, condition, device, z = None, bs = None):
    """Sample from the generator. Given a certain condition (the task id),
    we sample from the generator model a batch of replay data. This input of the 
    generator will be a noise vector (optional with a specific mean) and/or and
    additional task specific input. 

    Args:
        (....): See docstring of funct :func:`mnist.replay.train_replay.train`.
        condition: Condition (class/task) we want to sample from. Not to be 
        confused with the additional option that one can input a task specific 
        condition the replay model. 

    Returns:
        Batch of replay data from the generator, given a certain 
        condition / task id.
    """ 
    
    if z is None:
        # get the prior mean  
        if config.conditional_prior:
            cur_prior = config.priors[condition]
        else:
            cur_prior = torch.zeros((config.batch_size,
                                             config.latent_dim)).to(device)
        
        # sample normal gaussian and build noise vector
        eps = torch.randn_like(cur_prior)
        z = cur_prior + eps

    # get condition if given
    if config.conditional_replay:
        z = torch.cat([z, config.vae_conds[condition]], dim = 1)
    
    # cut for replay when we need the X_fake from all previous tasks need to sum
    # up the given batch_size such that batch_size(X_fake) == batch_size(X_real) 
    if bs is not None:
        z = z[:bs, :]

    # get weights from hnet
    if g_hnet is not None:
        weights_d = g_hnet.forward(condition)
    else:
        weights_d = None

    samples = gen.forward(z, weights_d)
    return torch.tanh(samples)

def train_gan_one_t(dhandler, dis, gen, g_hnet, device, config, writer,
                                                                 embd_list, t):
    """ Train the conditional MNIST GAN for one task.
    In this function the main training logic for this replay model is 
    implemented. After setting the optimizers for the discriminator/generator 
    and it's hypernetwork if applicable, a standart variational autoencoder 
    training scheme is implemented. To prevent the generator (its hypernetwork) 
    from forgetting, we add our hypernetwork regularisation term for all tasks 
    seen before ``t`` to the vae loss. 

    Args:
        (....): See docstring of function 
            :func:`mnist.replay.train_replay.train`.
        embd_list: Helper list of lists for embedding plotting.
        t: Task id to train.
    """

    print("Training GAN on data handler: ", t)

    # get lists for plotting embeddings
    d_embeddings, g_embeddings, d_embedding_history, g_embedding_history = \
                                                                    embd_list[:]        
    # set training_iterations if epochs are set
    if config.epochs == -1:
        training_iterations = config.n_iter
    else:
        assert(config.epochs > 0)
        training_iterations = config.epochs * \
        int(np.ceil(dhandler.num_train_samples / config.batch_size))
    
    # Here we adjust the number of training iterations when we train our replay 
    # method to replay every single class in a task given that condition. 
    # We need to adjust the training iterations such that we train every 
    # class in the task only a portion of the time we are given for the 
    # whole task:
    # Training_time_per_class = training_time_per_task / num_class_per_task
    # This is important to compare to related work, as they set the training 
    # time per task which we now have to split up.
      
    if config.single_class_replay:
        training_iterations = int(training_iterations/config.out_dim)
    
    # if we want to start training the new task with the weights of the previous
    # task we have to set the start embedding for the new task to the embedding
    # of the previous task. 
    if config.embedding_reset == "old_embedding" and t > 0:
        if g_hnet is not None:
            last_emb = g_hnet.get_task_embs()[t-1].detach().clone()
            g_hnet.get_task_embs()[t].data = last_emb

    # Compute targets for the hnet before training. 
    if t > 0:
        if config.rp_beta > 0 and g_hnet is not None:
            targets_G = hreg.get_current_targets(t, g_hnet)
        else:
            targets_G = None
    
    ############
    # OPTIMIZERS 
    ############

    # discriminator optimizer
    dis_paras = dis.parameters()
    doptimizer = optim.Adam(dis_paras, lr=config.enc_lr,
                                betas=(0.9, 0.999))

    # discriminator optimizer (hnet or weights directly)
    if g_hnet is not None:
        g_paras = list(g_hnet.theta)
        if not config.dont_train_rp_embeddings:
            # Set the embedding optimizer only for the current task embedding.
            # Note that we could here continue training the old embeddings.
            g_emb_optimizer = optim.Adam([g_hnet.get_task_emb(t)], 
                                    lr=config.dec_lr_emb, betas=(0.9, 0.999))
        else:
            g_emb_optimizer = None         
    else:
        g_emb_optimizer = None       
        g_paras = gen.parameters()

    goptimizer = optim.Adam(g_paras, lr=config.dec_lr, 
                                 betas=(0.9, 0.999))

    calc_reg = config.rp_beta > 0 and t > 0 and g_hnet is not None
    
    for i in range(training_iterations):
        ### Test network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained net.
        if i % config.val_iter == 0:
            test(dis, gen, g_hnet, device, config, writer,  i, t)
            gen.train()
            dis.train()
            if g_hnet is not None:
                g_hnet.train()

        if i % 100 == 0:
            print('Training iteration: %d.' % i)

        if config.show_plots:
            if g_hnet is not None:
                if(not config.no_cuda):
                    g_embedding_history.append(g_hnet.get_task_emb(t).
                                            clone().detach().cpu().numpy())
                else:
                    g_embedding_history.append(g_hnet.get_task_emb(t).
                                                clone().detach().numpy())

        #######
        # DATA 
        #######
        real_batch = dhandler.next_train_batch(config.batch_size)
        X_real = dhandler.input_to_torch_tensor(real_batch[0], 
                                                    device, mode='train')
        #shift data in range [-1, 1] so we can tanh the output of G
        X_real = X_real*2 - 1.0

        ######################
        # TRAIN DISCRIMINATOR
        ######################

        # set gradients again to zero
        doptimizer.zero_grad()
        goptimizer.zero_grad()
        if g_emb_optimizer is not None:
            g_emb_optimizer.zero_grad()

        # Note that X_fake is not normalize between 0 and 1
        # but like in in https://github.com/Zackory/Kera
        # s-MNIST-GAN/blob/master/mnist_gan.py
        # inputs are shiftet between [-1, 1] and X_fake is put through tanh
        #X_fake = torch.tanh(X_fake)                        
        X_fake = sample(gen, g_hnet, config, t, device)

        fake = dis.forward(X_fake)
        real = dis.forward(X_real)

        # compute discriminator loss
        dloss = gan_helpers.dis_loss(real, fake, config.loss_fun)

        # compute gradients for discriminator and take gradient step
        dloss.backward()
        doptimizer.step()
        
        ######################
        # TRAIN GENERATOR
        ######################

        # set gradients again to zero
        goptimizer.zero_grad()
        doptimizer.zero_grad()
        if g_emb_optimizer is not None:
            g_emb_optimizer.zero_grad()
        
        X_fake = sample(gen, g_hnet, config, t, device)
        fake = dis.forward(X_fake)

        # compute generator loss
        gloss = gan_helpers.gen_loss(fake, config.loss_fun)
        
        gloss.backward(retain_graph=calc_reg,create_graph=calc_reg and \
                           config.backprop_dt)

        # compute hypernet reg loss and fix embedding->change current embs
        if calc_reg:
            if config.no_lookahead:
                dTheta = None
            else:
                dTheta = opstep.calc_delta_theta(goptimizer,
                    config.use_sgd_change, lr=config.dec_lr,
                    detach_dt=not config.backprop_dt)

            gloss_reg =  config.rp_beta *hreg.calc_fix_target_reg(g_hnet, t,
                        targets=targets_G, mnet=gen, dTheta=dTheta, dTembs=None)
            gloss_reg.backward()
        else:
            gloss_reg = 0

        # compute gradients for generator and take gradient step
        goptimizer.step()
        if g_hnet is not None and not config.dont_train_rp_embeddings:
            g_emb_optimizer.step()
        
        # Visualization of current progress in tensorboard
        if i % config.plot_update_steps == 0 and i > 0 and config.show_plots:
            if d_embedding_history is not None:
                d_embedding_cut =  np.asarray(d_embedding_history[2:])
            else:
                d_embedding_cut = None
            if g_embedding_history is not None:
                g_embedding_cut = np.asarray(g_embedding_history[2:])
            else:
                g_embedding_cut = None
            _viz_training(X_real, X_fake, g_embeddings, d_embeddings, 
                g_embedding_cut, d_embedding_cut,
                writer, i, config, title="train_cond_" + str(t))

        # track some training statistics
        writer.add_scalar('train/gen_loss_%d' % (t), gloss + gloss_reg, i)
        writer.add_scalar('train/dloss_all_%d' % (t), dloss, i)
        writer.add_scalar('train/dis_accuracy_%d' % (t), 
                           gan_helpers.accuracy(real, fake, config.loss_fun), i)
        if config.rp_beta > 0:
            writer.add_scalar('train/g_hnet_loss_reg_%d' % (t), gloss_reg, i)
            writer.add_scalar('train/g_loss_only_%d' % (t), gloss, i)

    test(dis, gen, g_hnet, device, config, writer, config.n_iter, t)
                                                       
if __name__ == '__main__':
    print('Use "train_replay.py --replay_method gan" to train a replay GAN ' + 
          'with hypernetworks.')
