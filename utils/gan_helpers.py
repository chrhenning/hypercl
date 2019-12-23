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
#
# @title          :utils/gan_helpers.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :12/17/2019
# @version        :1.0
# @python_version :3.6.9
"""
A collection of helper functions that are usefull and general for GAN training,
e.g., several GAN losses.
"""
import torch
import torch.nn.functional as F

def dis_loss(logit_real, logit_fake, loss_choice):
    """Compute the loss for the discriminator.

    Note, only the discriminator weights should be updated using this loss.

    Args:
        logit_real: Outputs of the discriminator after seeing real samples.

            .. note::
                We assume a linear output layer.
        logit_fake: Outputs of the discriminator after seeing fake samples.

            .. note::
                We assume a linear output layer.
        loss_choice (int): Define what loss function is used to train the GAN.
            Note, the choice of loss function also influences how the output
            of the discriminator network  if reinterpreted or squashed (either
            between ``[0,1]`` or an arbitrary real number).

            The following choices are available.

            - ``0``: Vanilla GAN (Goodfellow et al., 2014). Non-saturating
              loss version. Note, we additionally apply one-sided label
              smoothing for this loss.
            - ``1``: Traditional LSGAN (Mao et al., 2018). See eq. 14 of
              the paper. This loss corresponds to a parameter
              choice :math:`a=0`, :math:`b=1` and :math:`c=1`.
            - ``2``: Pearson Chi^2 LSGAN (Mao et al., 2018). See eq. 13.
              Parameter choice: :math:`a=-1`, :math:`b=1` and :math:`c=0`.
            - ``3``: Wasserstein GAN (Arjovski et al., 2017).

    Returns:
        The discriminator loss.
    """
    if loss_choice == 0: # Vanilla GAN
        # We use the binary cross entropy.
        # Note, we use one-sided label-smoothing.
        fake = torch.sigmoid(logit_fake)
        real = torch.sigmoid(logit_real)
        r_loss = F.binary_cross_entropy(real, 0.9*torch.ones_like(real))
        f_loss = F.binary_cross_entropy(fake, torch.zeros_like(fake))

    elif loss_choice == 1: # Traditional LSGAN
        r_loss = F.mse_loss(logit_real, torch.ones_like(logit_real))
        f_loss = F.mse_loss(logit_fake, torch.zeros_like(logit_fake))

    elif loss_choice == 2: # Pearson Chi^2 LSGAN
        r_loss = F.mse_loss(logit_real, torch.ones_like(logit_real))
        f_loss = F.mse_loss(logit_fake, -torch.ones_like(logit_fake))
    else: # WGAN
        r_loss = -logit_real.mean()
        f_loss = logit_fake.mean()
    
    return (r_loss + f_loss)

def gen_loss(logit_fake, loss_choice):
    """Compute the loss for the generator.

    Args:
        (....): See docstring of function :func:`dis_loss`.

    Returns:
        The generator loss.
    """
    if loss_choice == 0: # Vanilla GAN
        # We use the -log(D) trick.
        fake = torch.sigmoid(logit_fake)
        return F.binary_cross_entropy(fake, torch.ones_like(fake))

    elif loss_choice == 1: # Traditional LSGAN
        return F.mse_loss(logit_fake, torch.ones_like(logit_fake))

    elif loss_choice == 2: # Pearson Chi^2 LSGAN
        return F.mse_loss(logit_fake, torch.zeros_like(logit_fake))

    else: # WGAN
        return -logit_fake.mean()

def accuracy(logit_real, logit_fake, loss_choice):
    """The accuracy of the discriminator.

    It is computed based on the assumption that values greater than a threshold
    are classified as real.

    Note, the accuracy measure is only well defined for the Vanilla GAN.
    Though, we just look at generally preferred value ranges and generalize
    the concept of accuracy to the other GAN formulations using the
    following thresholds:

    - ``0.5`` for Vanilla GAN and Traditional LSGAN
    - ``0`` for Pearson Chi^2 LSGAN and WGAN.

    Args:
        (....): See docstring of function :func:`dis_loss`.

    Returns:
        The relative accuracy of the discriminator.
    """
    T = 0.5 if loss_choice < 2 else 0.0

    #if loss_choice == 0:
    #    fake = torch.sigmoid(logit_fake)
    #    real = torch.sigmoid(logit_real)

    # Note, values above 0 will be above 0.5 after being passed  through a
    # softmax. Therefore, we take the threshold 0 for logit activations, if the
    # logits are supposed to be passed through a softmax.
    T = 0 if loss_choice == 0 else T

    n_correct = (logit_real > T).float().sum() + (logit_fake <= T).float().sum()
    return n_correct / (logit_real.numel() + logit_fake.numel())

def concat_mean_stats(inputs):
    """Add mean statistics to discriminator input.

    GANs often run into mode collapse since the discriminator sees every
    sample in isolation. I.e., it cannot detect whether all samples in a batch
    do look alike.

    A simple way to allow the discriminator to have access to batch statistics
    is to simply concatenate the mean (across batch dimension) of all
    discriminator samples to each sample.

    Args:
        inputs: The input batch to the discriminator.

    Returns:
        The modified input batch.
    """
    stats = torch.mean(inputs, 0, keepdim=True)
    stats = stats.expand(inputs.size())
    return torch.cat([stats, inputs], dim=1)

if __name__ == '__main__':
    pass


