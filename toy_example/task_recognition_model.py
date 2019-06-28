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
@title           :toy_example/task_recognition_model.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :05/02/2019
@version         :1.0
@python_version  :3.6.8

This module can be used to adressing the problem of inferring the task to be
solved from the inputs to the main network.

Therefore, we use a VAE-like approach. The encoder is the recognition model,
that maps the input X to two output layers. One is a softmax layer that
determines the task that has been detected from the input. The other output
are the parameters of a distribution from which a latent embedding can be
sampled, that should encode the characteristics of the input (this is only
needed by the decoder for a proper reconstruction).
The decoder is the replay network. It is needed to avoid catastrophic forgetting
in the encoder. I.e., it is used to generate fake samples of old tasks by
inducing the task id and a sample from a prior distribution in the network. In
this way, the encoder can also be trained to not forget previous tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from toy_example.main_model import MainNetwork

class RecognitionNet(nn.Module):
    """The recognition network consists of an encoder and decoder. The encoder
    gets as input X (the same input as the main network) and has two outputs:
    a softmax layer named alpha, that can be used to determine the task inferred
    from the input; and a latent embedding nu_z, that we interpret as parameters
    of a normal distribution (mean and log-variances). Using the
    reparametrization trick, we can sample z.
    Both, alpha and z are going to be the input to the decoder. The decoder aims
    to reconstruct the input of the encoder.

    The network consists only of fully-connected layers. The decoder
    architecture is a mirrored version of the encoder architecture, except for
    the fact that the input to the decoder is z and not nu_z.

    Attributes (additional to base class):
        encoder_weights: All parameters of the encoder network.
        decoder_weights: All parameters of the decoder network.
        dim_alpha: Dimensionality of the softmax layer alpha.
        dim_z: Dimensionality of the latent embeddings z.
    """
    def __init__(self, n_in, n_tasks, dim_z=8, enc_layers=[10,10],
                 activation_fn=torch.nn.ReLU(), use_bias=True):
        """Initialize the network.

        Args:
            n_in: Input size (input dim of encoder and output dim of decoder).
            n_tasks: The maximum number of tasks to be detected (size of
                softmax layer).
            dim_z: Dimensionality of latent space z.
            enc_layers: A list of integers, each denoting the size of a hidden
                layer in the encoder. The decoder will have layer sizes in
                reverse order.
            activation_fn: The nonlinearity used in hidden layers. If None, no
                nonlinearity will be applied.
            use_bias: Whether layers may have bias terms.
        """
        super(RecognitionNet, self).__init__()

        self._n_alpha = n_tasks
        self._n_nu_z = 2 * dim_z
        self._n_z = dim_z

        ## Enoder
        encoder_shapes = MainNetwork.weight_shapes(n_in=n_in,
            n_out=self._n_alpha+self._n_nu_z, hidden_layers=enc_layers,
            use_bias=use_bias)
        self._encoder = MainNetwork(encoder_shapes, activation_fn=activation_fn,
                                    use_bias=use_bias, no_weights=False,
                                    dropout_rate=-1, verbose=False)
        self._weights_enc = self._encoder.weights

        ## Decoder
        decoder_shapes = MainNetwork.weight_shapes(n_in=self._n_alpha+self._n_z,
            n_out=n_in, hidden_layers=list(reversed(enc_layers)),
            use_bias=use_bias)
        self._decoder = MainNetwork(decoder_shapes, activation_fn=activation_fn,
                                    use_bias=use_bias, no_weights=False,
                                    dropout_rate=-1, verbose=False)
        self._weights_dec = self._decoder.weights

        ## Prior
        # Note, when changing the prior, one has to change the method
        # "prior_matching".
        self._mu_z = torch.zeros(dim_z)
        self._sigma_z = torch.ones(dim_z)

        n_params = np.sum([np.prod(p.shape) for p in self.parameters()])
        print('Constructed recognition model with %d parameters.' % n_params)

    @property
    def dim_alpha(self):
        """Getter for read-only attribute dim_alpha.

        Returns:
            Size of alpha layer.
        """
        return self._n_alpha

    @property
    def dim_z(self):
        """Getter for read-only attribute dim_z.

        Returns:
            Size of z layer.
        """
        return self._n_z

    @property
    def encoder_weights(self):
        """Getter for read-only attribute encoder_weights.

        Returns:
            A torch.nn.ParameterList.
        """
        return self._weights_enc

    @property
    def decoder_weights(self):
        """Getter for read-only attribute decoder_weights.

        Returns:
            A torch.nn.ParameterList.
        """
        return self._weights_dec

    def forward(self, x):
        """This function computes
            x_rec = decode(encode(x))

        Note, the function utilizes the class members "encode" and "decode".

        Args:
            x: The input to the "autoencoder".

        Returns:
            x_rec: The reconstruction of the input.
        """
        alpha, _, z = self.encode(x)
        x_rec = self.decode(alpha, z)

        return x_rec

    def encode(self, x, ret_log_alpha=False, encoder_weights=None):
        """Encode a sample x -> "recognize the task of x".
        
        Args:
            x: An input sample (from which a task should be inferred).
            ret_log_alpha (optional): Whether the log-softmax distribution of
                the output layer alpha should be returned as well.
            encoder_weights (optional): If given, these will be the parameters
                used in the encoder rather than the ones maintained object
                internally.

        Returns:
            alpha: The softmax output (task classification output).
            nu_z: The parameters of the latent distribution from which "z" is
                sampled (i.e., the actual output of the encoder besides alpha).
                Note, that these parameters are the cooncatenated means and
                log-variances of the latent distribution.
            z: A latent space embedding retrieved via the reparametrization
                trick.
            (log_alpha): The log softmax activity of alpha.
        """
        phi_e = None
        if encoder_weights is not None:
            phi_e = encoder_weights

        h = self._encoder.forward(x, weights=phi_e)

        h_alpha = h[:, :self._n_alpha]
        alpha = F.softmax(h_alpha, dim=1)

        params_z = h[:, self._n_alpha:]
        mu_z = params_z[:, :self._n_z]
        logvar_z = params_z[:, self._n_z:]

        std_z = torch.exp(0.5 * logvar_z)
        z = Normal(mu_z, std_z).rsample()

        if ret_log_alpha:
            return alpha, params_z, z, F.log_softmax(h_alpha, dim=1)

        return alpha, params_z, z

    def decode(self, alpha, z, decoder_weights=None):
        """Decode a latent representation back to a sample.
        If alpha is a 1-hot encoding denoting a specific task and z are latent
        space samples, the decoding can be seen as "replay" of task samples.

        Args:
            alpha: See return value of method "encode".
            z: See return value of method "encode".
            decoder_weights (optional): If given, these will be the parameters
                used in the decoder rather than the ones maintained object
                internally.

        Returns:
            x_dec: The decoded sample.
        """
        phi_d = None
        if decoder_weights is not None:
            phi_d = decoder_weights

        x_dec = self._decoder.forward(torch.cat([alpha, z], dim=1),
                                      weights=phi_d)

        return x_dec

    def prior_samples(self, batch_size):
        """Obtain a batch of samples from the prior for the latent space z.

        Args:
            batch_sizeze: Number of samples to acquire.

        Returns:
            A torch tensor of samples.
        """
        return Normal(self._mu_z, self._sigma_z).rsample([batch_size])

    def prior_matching(self, nu_z):
        """Compute the prior matching term between the Gaussian described by
        the parameters "nu_z" and a standard normal distribution N(0, I).

        Args:
            nu_z: Part of the encoder output.

        Returns:
            The value of the prior matching loss.
        """
        mu_z = nu_z[:, :self._n_z]
        logvar_z = nu_z[:, self._n_z:]

        var_z = logvar_z.exp()

        return -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - var_z)

    @staticmethod
    def task_cross_entropy(log_alpha, target):
        """A call to pytorch "nll_loss".

        Args:
            log_alpha: The log softmax activity of the alpha layer.
            target: A vector of task ids.

        Returns:
            Cross-entropy loss
        """
        return F.nll_loss(log_alpha, target)

    @staticmethod
    def reconstruction_loss(x, x_rec):
        """A call to pytorch "mse_loss"

        Args:
            x: An input sample.
            x_rec: The reconstruction provided by the recognition AE when seeing
                input "x".

        Returns:
            The MSE loss between x and x_rec.
        """
        return F.mse_loss(x, x_rec)

if __name__ == '__main__':
    pass


