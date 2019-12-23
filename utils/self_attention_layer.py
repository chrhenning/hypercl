#!/usr/bin/env python3
# Copyright 2018 Cheonbok Park
"""
@title           :utils/self_attention_layer.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :02/21/2019
@version         :1.0
@python_version  :3.6.6

This function was copied from 

    https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py

It was written by Cheonbok Park. Unfortunately, no license was visibly
provided with this code.

Note, that we use this code WITHOUT ANY WARRANTIES.

The code was slightly modified to fit our purposes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.misc import init_params

class SelfAttnLayer(nn.Module):
    """Self-Attention Layer

    This type of layer was proposed by:

        Zhang et al., "Self-Attention Generative Adversarial Networks", 2018
        https://arxiv.org/abs/1805.08318

    The goal is to capture global correlations in convolutional networks (such
    as generators and discriminators in GANs).
    """
    def __init__(self, in_dim, use_spectral_norm):
        """Initialize self-attention layer.

        Args:
            in_dim: Number of input channels (C).
            use_spectral_norm: Enable spectral normalization for all 1x1 conv.
                layers.
        """
        super(SelfAttnLayer,self).__init__()
        self.channel_in = in_dim

        # 1x1 convolution to generate f(x).
        self.query_conv = nn.Conv2d(in_channels=in_dim ,
                                    out_channels=in_dim // 8, kernel_size=1)
        # 1x1 convolution to generate g(x).
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8,
                                  kernel_size=1)
        # 1x1 convolution to generate h(x).
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
                                    kernel_size=1)
        # This parameter is on purpose initialized to be zero as described in
        # the paper.
        self.gamma = nn.Parameter(torch.zeros(1))

        # Spectral normalization is used in the original implementation:
        #   https://github.com/brain-research/self-attention-gan
        # Note, the original implementation also appears to use an additional
        # (fourth) 1x1 convolution to postprocess h(x) * beta before adding it
        # to the input tensor. Though, the reason is not fully obvious to me and
        # seems to be not mentioned in the paper.
        if use_spectral_norm:
            self.query_conv = nn.utils.spectral_norm(self.query_conv)
            self.key_conv = nn.utils.spectral_norm(self.key_conv)
            self.value_conv = nn.utils.spectral_norm(self.value_conv)

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, ret_attention=False):
        """Compute and apply attention map to mix global information into local
        features.

        Args:
            x: Input feature maps (shape: B x C x W x H).
            ret_attention (optional): If the attention map should be returned
                as an additional return value.

        Returns:
            (tuple): Tuple (if ``ret_attention`` is ``True``) containing:

            - **out**: gamma * (self-)attention features + input features.
            - **attention**: Attention map, shape: B X N X N (N = W * H).
        """
        m_batchsize, C, width, height = x.size()

        # Compute f(x)^T, shape: B x N x C//8.
        proj_query  = self.query_conv(x).view(m_batchsize,-1, width*height).\
            permute(0,2,1)
        # Compute g(x), shape: B x C//8 x N.
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height)
        energy =  torch.bmm(proj_query, proj_key) # f(x)^T g(x)
        # We compute the softmax per column of "energy" -> columns should sum
        # up to 1.
        attention = self.softmax(energy) # shape: B x N x N
        # Compute h(x), shape: B x C x N.
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        # Compute h(x) * beta (equation 2 in the paper).
        # FIXME I am sure that taking the tranpose of "attention" is wrong, as
        # the columns (not rows) of "attention" sum to 1.
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        if ret_attention:
            return out, attention
        return out

class SelfAttnLayerV2(nn.Module):
    """Self-Attention Layer with weights maintained seperately. Hence, this
    class should have the exact same behavior as "SelfAttnLayer" but the weights
    are maintained independent of the preimplemented PyTorch modules, which
    allows more flexibility (e.g., generating weights by a hypernet or modifying
    weights easily).

    This type of layer was proposed by:

        Zhang et al., "Self-Attention Generative Adversarial Networks", 2018
        https://arxiv.org/abs/1805.08318

    The goal is to capture global correlations in convolutional networks (such
    as generators and discriminators in GANs).

    Attributes:
        weight_shapes: The shapes of all parameter tensors in this layer
            (value of attribute is independent of whether "no_weights" was
            set in the constructor).
        weights: A list of parameter tensors (all parameters in this layer).
    """
    def __init__(self, in_dim, use_spectral_norm, no_weights=False,
                 init_weights=None):
        """Initialize self-attention layer.

        Args:
            in_dim: Number of input channels (C).
            use_spectral_norm: Enable spectral normalization for all 1x1 conv.
                layers.
            no_weights: If set to True, no trainable parameters will be
                constructed, i.e., weights are assumed to be produced ad-hoc
                by a hypernetwork and passed to the forward function.
            init_weights (optional): This option is for convinience reasons.
                The option expects a list of parameter values that are used to
                initialize the network weights. As such, it provides a
                convinient way of initializing a network with a weight draw
                produced by the hypernetwork.
                See attribute "weight_shapes" for the format in which parameters
                should be passed.
        """
        super(SelfAttnLayerV2,self).__init__()
        assert(not no_weights or init_weights is None)
        if use_spectral_norm:
            raise NotImplementedError('Spectral norm not yet implemented ' +
                                      'for this layer type.')

        self.channel_in = in_dim

        self.softmax  = nn.Softmax(dim=-1)

        # 1x1 convolution to generate f(x).
        query_dim = [in_dim // 8, in_dim, 1, 1]
        # 1x1 convolution to generate g(x).
        key_dim = [in_dim // 8, in_dim, 1, 1]
        # 1x1 convolution to generate h(x).
        value_dim = [in_dim, in_dim, 1, 1]
        gamma_dim = [1]
        self._weight_shapes = [query_dim, [query_dim[0]],
                               key_dim, [key_dim[0]],
                               value_dim, [value_dim[0]],
                               gamma_dim
                              ]

        if no_weights:
            self._weights = None
            return

        ### Define and initialize network weights.
        self._weights = nn.ParameterList()

        for i, dims in enumerate(self._weight_shapes):
            self._weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))

        if init_weights is not None:
            assert(len(init_weights) == len(self._weight_shapes))
            
            for i in range(len(init_weights)):
                assert(np.all(np.equal(list(init_weights[i].shape),
                                       list(self._weights[i].shape))))
                self._weights[i].data = init_weights[i]
        else:
            for i in range(0, len(self._weights)-1, 2):
                init_params(self._weights[i], self._weights[i+1])
            # This gamma parameter is on purpose initialized to be zero as
            # described in the paper.
            nn.init.constant_(self._weights[-1], 0)

    @property
    def weight_shapes(self):
        """Getter for read-only attribute weight_shapes."""
        return self._weight_shapes

    @property
    def weights(self):
        """Getter for read-only attribute weights.

        Returns:
            A torch.nn.ParameterList or None, if this network has no weights.
        """
        return self._weights

    def forward(self, x, ret_attention=False, weights=None, dWeights=None):
        """Compute and apply attention map to mix global information into local
        features.

        Args:
            x: Input feature maps (shape: B x C x W x H).
            ret_attention (optional): If the attention map should be returned
                as an additional return value.
            weights: List of weight tensors, that are used as layer parameters.
                If "no_weights" was set in the constructor, then this parameter
                is mandatory.
                Note, when provided, internal parameters are not used.
            dWeights: List of weight tensors, that are added to "weights" (the
                internal list of parameters or the one given via the option
                "weights"), when computing the output of this network.

        Returns:
            (tuple): Tuple (if ``ret_attention`` is ``True``) containing:

            - **out**: gamma * (self-)attention features + input features.
            - **attention**: Attention map, shape: B X N X N (N = W * H).
        """
        if self._weights is None and weights is None:
            raise Exception('Layer was generated without internal weights. ' +
                            'Hence, "weights" option may not be None.')

        if weights is None:
            weights = self.weights
        else:
            assert(len(weights) == len(self.weight_shapes))

        if dWeights is not None:
            assert(len(dWeights) == len(self.weight_shapes))

            new_weights = []
            for i, w in enumerate(weights):
                new_weights.append(w + dWeights[i])
            weights = new_weights

        m_batchsize, C, width, height = x.size()

        # Compute f(x)^T, shape: B x N x C//8.
        proj_query = F.conv2d(x, weights[0], bias=weights[1]). \
            view(m_batchsize,-1, width*height).permute(0,2,1)
        # Compute g(x), shape: B x C//8 x N.
        proj_key = F.conv2d(x, weights[2], bias=weights[3]). \
            view(m_batchsize, -1, width*height)
        energy =  torch.bmm(proj_query, proj_key) # f(x)^T g(x)
        # We compute the softmax per column of "energy" -> columns should sum
        # up to 1.
        attention = self.softmax(energy) # shape: B x N x N
        # Compute h(x), shape: B x C x N.
        proj_value = F.conv2d(x, weights[4], bias=weights[5]). \
            view(m_batchsize, -1, width*height)

        # Compute h(x) * beta (equation 2 in the paper).
        # FIXME I am sure that taking the tranpose of "attention" is wrong, as
        # the columns (not rows) of "attention" sum to 1.
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)

        out = weights[6] * out + x

        if ret_attention:
            return out, attention
        return out

if __name__ == '__main__':
    pass


