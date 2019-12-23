#!/usr/bin/env python3
# Copyright 2019 Markus Marks
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
# @title           :plotting.py
# @author          :mm
# @contact         :marksm@ethz.ch
# @created         :07/17/2019    
# @version         :1.0
# @python_version  :3.7.3

"""
Plotting functions to help visualise the continual learning of GANs and VAEs
with hypernetworks. In particular we can the visualise the training 
trajectories of two dimensional embeddings during training or sampled replay 
data from decoders or generators.
"""

# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

def _viz_init(dhandlers, enc_hnet, dec_hnet, writer, config, lims = 8):
    """ Initial visualization of hypernetwork embeddings and datasets.
        
        Args:
            dhandlers: data handlers
            enc_hnet: encoder hypernetwork
            dec_hnet: decoder hypernetwork
            writer: tensorboard writer
            config: global config file
            lims: x/y plot limits for scatterplots

        Returns:
            enc_embeddings: intiial encoder embeddings for further plotting
            dec_embeddings: initial decoder embeddings for further plotting
    """
    
    if dec_hnet is not None:
        dec_embeddings = []
        for emb in dec_hnet.get_task_embs():
            if(not config.no_cuda):
                dec_embeddings.append(emb.cpu().detach().numpy())
            else:
                dec_embeddings.append(emb.detach().numpy())
        dec_embeddings = np.asarray(dec_embeddings)
    else:
        dec_embeddings = None

    if enc_hnet is not None:
        enc_embeddings = []
        for emb in enc_hnet.get_task_embs():
            if(not config.no_cuda):
                enc_embeddings.append(emb.cpu().detach().numpy())
            else:
                enc_embeddings.append(emb.detach().numpy())
        enc_embeddings = np.asarray(enc_embeddings)
    else:
        enc_embeddings = None


    data_batch = dhandlers[0].next_test_batch(config.batch_size)
    X = dhandlers[0].input_to_torch_tensor(data_batch[0], 'cpu',
        mode='inference')
    figure = _plotImages(X, config)

    writer.add_figure('overall', figure)
    
    if dec_hnet is not None:
        fig = _scatterPlotData([dec_embeddings], config,['blue'])
        writer.add_figure('decoder embeddings', fig)
    
    if enc_hnet is not None:
        fig = _scatterPlotData([enc_embeddings], config,['blue'])
        writer.add_figure('encoder embeddings', fig)

    return enc_embeddings, dec_embeddings

def _plotImages(X, config, cur_bs = None):
    """ Helper function to plot MNIST images.

        Args:
            X: Images to plot (can be fake or real).
            config: The command line arguments.
    """

    add = config.padding*2

    if not config.no_cuda:
        X = X.cpu().detach().numpy()
    else:
        X = X.detach().numpy()
    if cur_bs is None:
        X = X.reshape(config.batch_size, 28 + add, 28 + add)
    else:
        X = X.reshape(cur_bs, 28 + add, 28 + add)
    num_plots = 4

    fig, ax = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3): 
            ax[i, j].imshow(X[i*num_plots+j], 
                                        interpolation='nearest',cmap='gray_r')
            ax[i, j].axis('off')
    return fig

def _scatterPlotData(datasets, config, colors, lims=None):
    """ Helper function to generate Scatterplots.

        Args:
            datasets: datasets to be scatter-plotted
            config: The command line arguments.
            colors: colors for each dataset to be plotted, order with
                respect to datasets
            lims: x/y plot limits for scatterplots
    """
    fig, ax = plt.subplots()
    if(lims):
        ax.set_xlim([-lims, lims])
        ax.set_ylim([-lims, lims])
    # add colors
    for idx, dataset in enumerate(datasets):
        if(type(dataset)==np.ndarray):
            plt.scatter(dataset[:,0],
                    dataset[:,1], color=colors[idx])
        elif(not config.no_cuda):
            plt.scatter(dataset.cpu().detach().numpy()[:,0],
                    dataset.cpu().detach().numpy()[:,1], color=colors[idx])
        else:
            plt.scatter(dataset.detach().numpy()[:,0],
                    dataset.detach().numpy()[:,1], color=colors[idx])
    return fig

def _viz_training(data_real, x_fake, enc_embeddings, dec_embeddings,
        enc_embedding_history, dec_embedding_history,
        writer, step, config, lims=8, title='train'):
    """ Visualize the trianing process.

        Args:
            data_real: real data
            x_fake: g generated fake data
            enc_embeddings: current encoder embedding
            dec_embeddings: current decoder embedding
            enc_embedding_history: history of embeddings of the encoder
            dec_embedding_history: history of embeddings of the decoder
            writer: tensorboard writer
            step: current step/iteration
            config: global config
            lims: x/y plot limits for scatter plots
    """
    
    # plot fake data
    fig = _plotImages(data_real, config)
    writer.add_figure(title + '_real', fig, global_step=step)

    # plot real data
    fig = _plotImages(x_fake, config)
    writer.add_figure(title +'_fake', fig, global_step=step)

    # plot embedding history with current embedding
    #TODO: restart scale for each mode
    
    fig, ax = plt.subplots()
    if dec_embeddings is not None:
        plt.scatter(dec_embeddings[:,0],
                dec_embeddings[:,1], color = 'red')

    if dec_embedding_history is not None:
        t = np.log(np.arange(1, len(dec_embedding_history)+ 1))  
        plt.scatter(dec_embedding_history[:,0], dec_embedding_history[:,1], c=t)
    
    writer.add_figure(title[:5] + '_decoder_embeddings', 
                                                        fig, global_step=step)

    fig, ax = plt.subplots()
    if enc_embeddings is not None:
        plt.scatter(enc_embeddings[:,0],
                enc_embeddings[:,1], color = 'red')

    if enc_embedding_history is not None:
        t = np.log(np.arange(1, len(enc_embedding_history)+ 1))
        plt.scatter(enc_embedding_history[:,0], enc_embedding_history[:,1], c=t)
    
    writer.add_figure(title[:5] + '_encoder_embeddings',
                                                        fig, global_step=step)
