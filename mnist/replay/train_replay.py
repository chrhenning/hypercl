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
# @title           :train.py
# @author          :jvo
# @contact         :voswaldj@ethz.ch
# @created         :07/10/2019
# @version         :1.0
# @python_version  :3.6.8

"""
Continual learning of MNIST VAE with hypernetworks
---------------------------------------------------

An implementation of a simple fully-connected MNIST VAE realized through
a hypernetwork, i.e., a hypernetwork that produces the weights of the decoder.
"""

# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import os
from warnings import warn
import numpy as np

from mnist import train_utils
from mnist.replay import train_utils_replay
from mnist.replay import train_args_replay
from mnist.plotting import _viz_init, _viz_training, _plotImages
from mnist.replay.train_gan import train_gan_one_t

import utils.hnet_regularizer as hreg
import utils.optim_step as opstep


def test(enc, dec, d_hnet, device, config, writer, train_iter=None, 
                                                                condition=None):
    """ Test the MNIST VAE - here we only sample from a fixed noise to compare
    images qualitatively. One should also keep track of the reconstruction 
    error of e.g. a test set.

    Args:
        (....): See docstring of function :func:`train`.
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
    enc.eval()
    dec.eval()
    if d_hnet is not None:
        d_hnet.eval()

    with torch.no_grad():
        # iterate over all conditions
        for m in range(condition + 1):
            # Get pre training saved noise
            z = config.test_z[m]
            reconstructions = sample(dec, d_hnet, config, m, device, z = z)
            if config.show_plots:
                fig_real = _plotImages(reconstructions, config)
                writer.add_figure('test_cond_' + str(m) + 
                                    '_sampled_after_'+str(condition), fig_real, 
                                                global_step=train_iter)
                if train_iter == config.n_iter:
                    writer.add_figure('test_cond_final_' + str(m) + 
                                    '_sampled_after_'+str(condition), fig_real, 
                                                global_step=train_iter)
            # TODO write test reconstrunction error           

def sample(dec, d_hnet, config, condition, device, z = None, bs = None):
    """Sample from the decoder. Given a certain condition (the task id),
    we sample from the decoder model a batch of replay data. This input of the 
    decoder will be a noise vector (optional with a specific mean) and/or and
    additional task specific input. 

    Args:
        (....): See docstring of function :func:`train`.
        condition: Condition (class/task) we want to sample from. Not to be 
        confused with the additional option that one can input a task specific 
        condition the replay model. 

    Returns:
        Batch of replay data from the decoder, given a certain 
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
    if d_hnet is not None:
        weights_d = d_hnet.forward(condition)
    else:
        weights_d = None
    samples = dec.forward(z, weights_d)

    return torch.sigmoid(samples)

def init_plotting_embedding(dhandlers, d_hnet, writer, config):
    """ This is a helper function to get lists to plot embedding histories.

    Args:
        (....): See docstring of function :func:`train`.
    Returns:
        List of lists for embedding plots during training.
    """

    # initial visualization and setting up training viz
    if config.show_plots:
        _, dec_embs = _viz_init(dhandlers, None, d_hnet, writer, config)
        if d_hnet is not None:
            dec_embs_history = []
            if(not config.no_cuda):
                dec_embs_history.append(d_hnet.get_task_emb(0).
                                                        cpu().detach().numpy())
            else:
                dec_embs_history.append(d_hnet.get_task_emb(0).
                                                        detach().numpy())
        else:
            dec_embs_history = None

        return [None, dec_embs, None, dec_embs_history]
    else:
        return [None, None, None, None]

def reparameterize(mu, logvar):
    """Reparameterize encoder output for vae loss. Code from
        https://github.com/pytorch/examples/blob/master/vae/main.py#L48

    Args:
        mu: Output of encoder parameterising the mean of the Gaussian.
        logvar: Output of the encoder that get transformed into the
            variance to be used for the reparameterization trick below.
        eps: Use epsilon already drawn to reduce variance

    Returns:
        Sample from the Gaussian through the reparameterization trick.
    """
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def compute_kld(mu, logvar, config, t):
    """Compute the kullback-leibler divergence between normal gaussian around
    zero or mu_prior and a gaussian with parameters mu, logvar.

    Args:
        mu: Outputs of the encoder, mean of the VAE latent Gaussian.
        logvar: Outputs of the encoder, logvar of the VAE latent Gaussian.
        config: Command-line arguments.
        t: task id.            
    Returns:
        LKD between gausian with parameters by encoder and prior.
    """

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # add prior matching loss
    if config.conditional_prior:
        cur_prior = config.priors[t]
    else:
        cur_prior = 0
    kld = -0.5 * torch.sum(1 + logvar - (mu - cur_prior).pow(2) - \
                            logvar.exp(), dim =1)
    # average kl by input dim (to compare to related work, see
    # https://github.com/GMvandeVen/continual-learning/blob/master/train.py)

    kld = torch.mean(kld) / config.input_dim
    return kld

def train_vae_one_t(dhandler, enc, dec, d_hnet, device, config, writer, 
                                                                  embd_list, t):
    """ Train the conditional MNIST VAE for one task.
    In this function the main training logic for this replay model is 
    implemented. After setting the optimizers for the encoder/decoder and it's
    hypernetwork if applicable, a standart variational autoencoder training
    scheme is implemented. To prevent the decoder (its hypernetwork) from 
    forgetting, we add our hypernetwork regularisation term for all tasks 
    seen before ``t`` to the vae loss. 

    Args:
        (....): See docstring of function :func:`train`.
        embd_list: Helper list of lists for embedding plotting.
        t: Task id that will be trained.

    """
    
    # set to training mode 
    enc.train()
    dec.train()
    if d_hnet is not None:
        d_hnet.train()

    # reset data handler
    print("Training VAE on data handler: ", t)

    # get lists for plotting embeddings
    enc_embs, dec_embs, enc_embs_history, dec_embs_history = embd_list[:]        
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
        if d_hnet is not None:
            last_emb = d_hnet.get_task_embs()[t-1].detach().clone()
            d_hnet.get_task_embs()[t].data = last_emb

    # Compute targets for the hnet before training. 
    if t > 0:
        if config.rp_beta > 0 and d_hnet is not None:
            targets_D = hreg.get_current_targets(t, d_hnet)
        else:
            targets_D = None
    
    ############
    # OPTIMIZERS 
    ############

    # encoder optimizer
    e_paras = enc.parameters()
    eoptimizer = optim.Adam(e_paras, lr=config.enc_lr,
                                betas=(0.9, 0.999))

    # decoder optimizer (hnet or weights directly)
    if d_hnet is not None:
        d_paras = list(d_hnet.theta)
        if not config.dont_train_rp_embeddings:
            # Set the embedding optimizer only for the current task embedding.
            # Note that we could here continue training the old embeddings.
            d_emb_optimizer = optim.Adam([d_hnet.get_task_emb(t)], 
               lr=config.dec_lr_emb, betas=(0.9, 0.999))
        else:
            d_emb_optimizer = None         
    else:
        d_emb_optimizer = None       
        d_paras = dec.parameters()

    doptimizer = optim.Adam(d_paras, lr=config.dec_lr, 
                                 betas=(0.9, 0.999))

    calc_reg = config.rp_beta > 0 and t > 0 and d_hnet is not None

    ###########
    # TRAINING 
    ###########

    for i in range(training_iterations):
        ### Test network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained net.
        if i % config.val_iter == 0:
            test(enc, dec, d_hnet, device, config, writer, i, t)
            enc.train()
            dec.train()
            if d_hnet is not None:
                d_hnet.train()

        if i % 100 == 0:
            print('Training iteration: %d.' % i)

        # Some code for plotting. 
        # We want to visualize the hnet embedding trajectories. 
        if config.show_plots:
            if d_hnet is not None:
                if(not config.no_cuda):
                    dec_embs_history.append(d_hnet.get_task_emb(t).
                                            clone().detach().cpu().numpy())
                else:
                    dec_embs_history.append(d_hnet.get_task_emb(t).
                                                clone().detach().numpy())

        #######
        # DATA 
        #######
        real_batch = dhandler.next_train_batch(config.batch_size)
        X_real = dhandler.input_to_torch_tensor(real_batch[0], device, 
                                                                   mode='train')

        # set gradients again to zero
        eoptimizer.zero_grad()
        doptimizer.zero_grad()
        if d_emb_optimizer is not None:
            d_emb_optimizer.zero_grad()
    
        ############################
        # KLD + RECONSTRUCTION 
        ############################

        # feed data through encoder
        mu_var = enc.forward(X_real)
        mu = mu_var[:,0: config.latent_dim]
        logvar= mu_var[:, config.latent_dim:2* config.latent_dim]
        
        # compute KLD
        kld = compute_kld(mu, logvar, config, t)

        # sample from encoder gaussian distribution
        dec_input = reparameterize(mu, logvar)
        reconstructions = sample(dec, d_hnet, config, t, device, z = dec_input)
        # average reconstruction error like this to compare to related work, see
        # https://github.com/GMvandeVen/continual-learning/blob/master/train.py

        x_rec_loss = F.binary_cross_entropy(reconstructions,
                                            X_real, reduction='none')
        x_rec_loss = torch.mean(x_rec_loss, dim=1)
        x_rec_loss = torch.mean(x_rec_loss)
        
        loss = x_rec_loss + kld

        ######################################################
        # HYPERNET REGULARISATION - CONTINUAL LEARNING METHOD
        ######################################################

        loss.backward(retain_graph=calc_reg,create_graph=calc_reg and \
                           config.backprop_dt)

        # compute hypernet loss and fix embedding -> change current embs
        if calc_reg:
            if config.no_lookahead:
                dTheta = None
            else:
                dTheta = opstep.calc_delta_theta(doptimizer,
                    config.use_sgd_change, lr=config.dec_lr,
                    detach_dt=not config.backprop_dt)
            dloss_reg = config.rp_beta* hreg.calc_fix_target_reg(d_hnet, t, 
                            targets=targets_D, 
                            mnet=dec, dTheta=dTheta, dTembs=None)
            dloss_reg.backward()                 
        else:
            dloss_reg = 0
        
        # compute gradients for generator and take gradient step
        doptimizer.step()
        eoptimizer.step()
        if d_hnet is not None and not config.dont_train_rp_embeddings:
            d_emb_optimizer.step()
        
        # Visualization of current progress in tensorboard
        if(i % config.plot_update_steps == 0 and i > 0 and config.show_plots):
            if dec_embs_history is not None:
                dec_embedding_cut =  np.asarray(dec_embs_history[2:])
            else:
                dec_embedding_cut = None
            if enc_embs_history is not None:
                enc_embedding_cut = np.asarray(enc_embs_history[2:])
            else:
                enc_embedding_cut = None
            _viz_training(X_real, reconstructions, enc_embs, 
                dec_embs, enc_embedding_cut, dec_embedding_cut,
                writer, i, config, title="train_cond_" + str(t))

        # track some training statistics
        writer.add_scalar('train/kld_%d' % (t), kld, i)
        writer.add_scalar('train/reconstruction_%d' % (t), x_rec_loss, i)
        writer.add_scalar('train/all_loss_%d' % (t), loss + dloss_reg, i)
        if config.rp_beta > 0:
            writer.add_scalar('train/d_hnet_loss_reg_%d' % (t), dloss_reg, i)

    test(enc, dec, d_hnet, device, config, writer, config.n_iter, t)

def train(dhandlers, enc, dec, d_hnet, device, config, writer):
    """ Train replay model in continual fashion on MNIST dataset.
    This is a helper function that loops over the range of tasks and 
    iteratively starts training the replay model on new tasks. 

    Args:
        dhandlers: The dataset handlers.
        enc: The model of the encoder network.
        dec. The model of the decoder network.
        d_hnet. The model of the decoder hyper network.
        device: Torch device (cpu or gpu).
        latent_sampler: An initialized distribution, we can sample from.
        config: The command line arguments.
        writer: The tensorboard summary writer.
    """

    print('Training the MNIST replay model ...')
    
    # get embedding lists for plotting
    embd_list = init_plotting_embedding(dhandlers, d_hnet, writer, config)
    # train the replay model task by task
    for t in range(config.num_embeddings):
        if config.replay_method == 'gan':
            train_gan_one_t(dhandlers[t], enc, dec, d_hnet, device, 
                                                   config, writer, embd_list, t)
        else:
            train_vae_one_t(dhandlers[t], enc, dec, d_hnet, device, 
                                                   config, writer, embd_list, t)
def run(config, train_system=True, only_train_replay=False, train_tandem=True):  
    """ Method to start training MNIST replay model. 
    Depending on the configurations, here we control the creation and 
    training of the different replay modules with their corresponding 
    hypernetworks.
        
    Args:
        config: The command line arguments.
        train_system: (optional) Set to false if we want this function 
            only to create config, networks and data_handlers for future 
            training. See :func:`mnist.train_splitMNIST.run` for a use case.
        only_train_replay: (optional) If this script will only be used to 
            train a replay model. Normally, we use this script in tandem 
            with an additional classifier that uses this replay model to 
            replay old tasks data.
        train_tandem: (optional) If we will use this script to train in 
            tandem i.e. in an alternating fashion with a classifier.
    Returns:
        (tuple): Tuple containing:
        (....): See docstring of function :func:`train`.
    """

    # if we want to train a classifier on single classes then we need a single
    # class replay method. This need not be the case otherwise i.e. we can 
    # have a single class replay method but train our classifier on the 
    # replay data (build out of multiple replayed conidtions) and the current
    # data at once.
    # single class replay only implemented for splitMNIST
    if config.single_class_replay:
        assert(config.experiment == "splitMNIST")

    if config.num_tasks > 100 and config.cl_scenario != 1:
        print("Attention: Replay model not tested for num tasks > 100")
        
    ### Setup environment
    device, writer = train_utils._setup_environment(config)

    ### Create tasks for split MNIST
    if config.single_class_replay:
        steps = 1
    else:
        steps = 2

    ### Create tasks for split MNIST
    if train_system == False and config.upper_bound == False:
        dhandlers =  None
    else:
        dhandlers = train_utils._generate_tasks(config, steps)

    ### Generate networks.
    if train_system == False:
        enc, dec, d_hnet =  None, None, None
    else:
        if config.rp_beta > 0:
            create_rp_hnet = True
        else:
            create_rp_hnet = False
        enc, dec, d_hnet = train_utils_replay.generate_replay_networks(config, 
                                              dhandlers, device, create_rp_hnet, 
                                            only_train_replay=only_train_replay)
        ### Generate task prioirs for latent space.
        priors = []
        test_z = []
        vae_conds = []
        
        ### Save some noise vectors for testing
        
        for t in range(config.num_embeddings):
            # if conditional prior create some task priors and save them
            if config.conditional_prior:
                mu = torch.zeros(( config.latent_dim)).to(device)
                nn.init.normal_(mu, mean=0, std=1.)
                mu = torch.stack([mu]*config.batch_size)
                mu.requires_grad = False
                priors.append(mu)
            else:
                mu = torch.zeros((config.batch_size, 
                                             config.latent_dim)).to(device)
                priors.append(None)

            ### Generate sampler for latent space.
            eps = torch.randn_like(mu)

            sample = mu + eps
            sample.requires_grad = False
            test_z.append(sample)
            
            # if vae has some conditional input, then either save hot-encodings
            # or some conditions from a gaussian
            if config.conditional_replay:
                vae_c = torch.zeros((config.conditional_dim)).to(device)
                if not config.not_conditional_hot_enc:
                    vae_c[t] = 1
                else:
                    nn.init.normal_(vae_c, mean=0, std=1.)
                vae_c = torch.stack([vae_c]*config.batch_size)
                vae_c.requires_grad = False
                vae_conds.append(vae_c)

        config.test_z = test_z
        config.priors = priors
        config.vae_conds = vae_conds
        if not train_tandem:
            ### Train the network.
            train(dhandlers, enc, dec, d_hnet, device, config, writer)

            ### Test network.
            test(enc, dec, d_hnet, device, config, writer)

    return dec, d_hnet, enc, dhandlers, device, writer, config 

if __name__ == '__main__':

    ### Get command line arguments.
    config = train_args_replay.parse_rp_cmd_arguments(mode='perm')

    ### run the scripts
    dec, d_hnet, enc, dhandlers, device, writer, config  = \
                        run(config, only_train_replay=True, train_tandem=False)   
    writer.close()
    print('Program finished successfully.')
