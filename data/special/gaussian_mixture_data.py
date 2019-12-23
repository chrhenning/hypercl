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
# @title           :gaussian_mixture_data.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :04/30/2019
# @version         :1.0
# @python_version  :3.6.8
"""
Gaussian Mixture Dataset
^^^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.special.gaussian_mixture_data` contains a toy dataset
consisting of input data drawn from a 2D Gaussian mixture distribution.

The dataset is inspired by the toy example provided in section 4.5 of
    https://arxiv.org/pdf/1606.00704.pdf

However, the mixture of Gaussians only determines the input domain x (which is
enough for a GAN dataset). Though, we also need to specify the output y.

For instance, each Gaussian bump could be the input domain of one task. Given
this input domain, the task would be to predict p(x), thus y = p(x).

In the case of small variances, the task can be detected from seeing the input x
alone. This allows us to predict task embeddings based on inputs, such that
there is no need to define the task embedding manually.
"""
import numpy as np
from scipy.stats import multivariate_normal
import itertools
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.spatial import cKDTree

from data.dataset import Dataset

# The default specfication of the dataset inputs is taken from
#   https://git.io/fjZlv
DEFAULT_MEANS = [np.array([i, j]) for i, j in
                 itertools.product(range(-4, 5, 2), range(-4, 5, 2))]
DEFAULT_VARIANCES = [0.05**2 * np.eye(len(mean)) for mean in DEFAULT_MEANS]

### Here are a few other configurations used in papers.
# https://arxiv.org/pdf/1611.02163.pdf
METZ_ANGLES = [i/8 * 2 * np.pi for i in range(8)]
METZ_MEANS = [np.array([2. * np.sin(a), 2. * np.cos(a)]) for a in METZ_ANGLES]
METZ_VARIANCES = [0.02**2 * np.eye(len(mean)) for mean in METZ_MEANS]

# https://arxiv.org/pdf/1612.02136.pdf
CHE_ANGLES = [(i+0.5)/6 * 2 * np.pi for i in range(6)]
CHE_MEANS = [np.array([5. * np.sin(a), 5. * np.cos(a)]) for a in CHE_ANGLES]
CHE_VARIANCES = [0.1**2 * np.eye(len(mean)) for mean in CHE_MEANS]

def get_gmm_tasks(means=DEFAULT_MEANS, covs=DEFAULT_VARIANCES, num_train=100,
                 num_test=100, map_functions=None, rseed=None):
    """Generate a set of data handlers (one for each task) of class
    :class:`GaussianData`.

    Args:
        means: The mean of each Gaussian.
        covs: The covariance matrix of each Gaussian.
        num_train: Number of training samples per task.
        num_test: Number of test samples per task.
        map_functions (optional): A list of "map_functions", one for each task.
        rseed: If ``None``, the current random state of numpy is used to
            generate the data. Otherwise, a new random state with the given seed
            is generated.

    Returns:
        (list): A list of objects of class :class:`GaussianData`.
    """
    assert(len(means) == len(covs))

    if map_functions is None:
        map_functions = [None] * len(means)
    else:
        assert(len(map_functions) == len(means))

    ret = []
    for i in range(len(means)):
        ret.append(GaussianData(mean=means[i], cov=covs[i], num_train=num_train,
            num_test=num_test, map_function=map_functions[i], rseed=rseed))

    return ret

class GaussianData(Dataset):
    """An instance of this class shall represent a regression task where the
    input samples :math:`x` are drawn from a Gaussian with given mean and
    variance.

    Due to plotting functionalities, this class only supports 2D inputs and
    1D outputs.

    Attributes:
        mean: Mean vector.
        cov: Covariance matrix.
    """
    def __init__(self, mean=np.array([0, 0]), cov=0.05**2 * np.eye(2),
                 num_train=100, num_test=100, map_function=None, rseed=None):
        """Generate a new dataset.

        The input data x for train and test samples will be drawn iid from the
        given Gaussian. Per default, the map function is the probability
        density of the given Gaussian: y = f(x) = p(x).

        Args:
            mean: The mean of the Gaussian.
            cov: The covariance of the Gaussian.
            num_train: Number of training samples.
            num_test: Number of test samples.
            map_function (optional): A function handle that receives input
                samples and maps them to output samples. If not specified, the
                density function will be used as map function.
            rseed: If None, the current random state of numpy is used to
                generate the data. Otherwise, a new random state with the given
                seed is generated.
        """
        super().__init__()

        if rseed is None:
            rand = np.random
        else:
            rand = np.random.RandomState(rseed)

        n_x = mean.size
        assert(n_x == 2) # Only required when using plotting functions.

        train_x = rand.multivariate_normal(mean, cov, size=num_train)
        test_x = rand.multivariate_normal(mean, cov, size=num_test)

        if map_function is None:
            map_function = lambda x : multivariate_normal.pdf(x, mean, cov). \
                reshape(-1, 1)

            # f(x) = p(x)
            train_y = map_function(train_x)
            test_y = map_function(test_x)
        else:
            train_y = map_function(train_x)
            test_y = map_function(test_x)

        # Specify internal data structure.
        self._data['classification'] = False
        self._data['sequence'] = False
        self._data['in_data'] = np.vstack([train_x, test_x])
        self._data['in_shape'] = [n_x]
        self._data['out_data'] = np.vstack([train_y, test_y])
        self._data['out_shape'] = [1]
        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train, num_train + num_test)

        self._mean = mean
        self._cov = cov
        self._map = map_function

    @property
    def mean(self):
        """Getter for read-only attribute :attr:`mean`."""
        return self._mean

    @property
    def cov(self):
        """Getter for read-only attribute :attr:`cov`."""
        return self._cov

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'GaussianInputData'

    def plot_samples(self, title, inputs, outputs=None, predictions=None,
                     num_samples_per_row=4, show=True, filename=None,
                     interactive=False, figsize=(10, 6)):
        """Plot samples belonging to this dataset.

        Note:
            Either ``outputs`` or ``predictions`` must be not ``None``!

        Args:
            title: The title of the whole figure.
            inputs: A 2D numpy array, where each row is an input sample.
            outputs (optional): A 2D numpy array of actual dataset targets.
            predictions (optional): A 2D numpy array of predicted output
                samples (i.e., output predicted by a neural network).
            num_samples_per_row: Maximum number of samples plotted
                per row in the generated figure.
            show: Whether the plot should be shown.
            filename (optional): If provided, the figure will be stored under
                this filename.
            interactive: Turn on interactive mode. We mainly
                use this option to ensure that the program will run in
                background while figure is displayed. The figure will be
                displayed until another one is displayed, the user closes it or
                the program has terminated. If this option is deactivated, the
                program will freeze until the user closes the figure.
                Note, if using the iPython inline backend, this option has no
                effect.
            figsize: A tuple, determining the size of the
                figure in inches.
        """
        assert(outputs is not None or predictions is not None)

        plt.figure(figsize=figsize)
        plt.title(title, size=20)
        if interactive:
            plt.ion()

        X1, X2, Y = self._get_function_vals()
        f = plt.contourf(X1, X2, Y)
        plt.colorbar(f)

        if outputs is not None:
            plt.scatter(inputs[:, 0], inputs[:, 1], edgecolors='b',
                label='Targets',
                facecolor=f.cmap(f.norm(outputs.squeeze())))
        if predictions is not None:
            plt.scatter(inputs[:, 0], inputs[:, 1], edgecolors='r',
                label='Predictions',
                facecolor=f.cmap(f.norm(predictions.squeeze())))
        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        if show:
            plt.show()

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Not implemented"""
        # We overwrote the plot_samples method, so there is no need to ever call
        # this method (it's just here because the baseclass requires its
        # existence).
        raise NotImplementedError('TODO implement')

    def plot_dataset(self):
        """Plot the whole dataset."""
        
        fig, ax = plt.subplots()
        train_x = self.get_train_inputs()
        train_y = self.get_train_outputs().squeeze()

        test_x = self.get_test_inputs()
        test_y = self.get_test_outputs().squeeze()

        X1, X2, Y = self._get_function_vals()
        heatmap = plt.contourf(X1, X2, Y)
        plt.colorbar(heatmap)

        #plt.scatter(train_x[:, 0], train_x[:, 1], edgecolors='r', label='Train',
        #            facecolors='none')
        #plt.scatter(test_x[:, 0], test_x[:, 1], edgecolors='b', label='Test',
        #            facecolors='none')

        # In case outputs might be noisy, we draw facecolors to match the
        # output value rather than drawing circles with no fill.
        plt.scatter(train_x[:, 0], train_x[:, 1], edgecolors='r', label='train',
                    facecolor=heatmap.cmap(heatmap.norm(train_y)))
        plt.scatter(test_x[:, 0], test_x[:, 1], edgecolors='b', label='test',
                    facecolor=heatmap.cmap(heatmap.norm(test_y)))

        plt.legend()
        plt.title('Gaussian Input Dataset')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

        return fig

    def _get_function_vals(self, grid_size=100):
        """Get real function values for a grid of equidistant x values in a
        range that covers the test and training data. These values can be used
        to plot the ground truth function.

        Args:
            grid_size: Width (or height) of the quadratic grid.

        Returns:
            X1, X2, Y: Three numpy arrays (2d) containing the corresponding x
                and y values. X1 and X2 follow the np.meshgrid definition.
        """
        train_x = self.get_train_inputs()
        test_x = self.get_test_inputs()

        mu = self._mean

        dx = max(np.abs(train_x - mu[None, :]).max(),
                 np.abs(test_x - mu[None, :]).max())
        dx = 1.05 * dx

        x1 = np.linspace(start=mu[0]-dx, stop=mu[0]+dx, num=grid_size)
        x2 = np.linspace(start=mu[1]-dx, stop=mu[1]+dx, num=grid_size)

        X1, X2 = np.meshgrid(x1, x2)

        X = np.vstack([X1.ravel(), X2.ravel()]).T

        Y = self._map(X).reshape(X1.shape)

        return X1, X2, Y

    def plot_predictions(self, predictions, label='Pred', show_train=True,
                         show_test=True):
        """Plot the dataset as well as predictions.

        Args:
            predictions: A tuple of x and y values, where the y values are
                computed by a trained regression network. Note, that x is
                supposed to be 2D numpy array, whereas y is a 1D numpy array.
            label: Label of the predicted values as shown in the legend.
            show_train: Show train samples.
            show_test: Show test samples.
        """
        train_x = self.get_train_inputs()
        train_y = self.get_train_outputs().squeeze()
        
        test_x = self.get_test_inputs()
        test_y = self.get_test_outputs().squeeze()

        X1, X2, Y = self._get_function_vals()
        f = plt.contourf(X1, X2, Y)
        plt.colorbar(f)

        if show_train:
            plt.scatter(train_x[:, 0], train_x[:, 1], edgecolors='r',
                label='Train', facecolor=f.cmap(f.norm(train_y.squeeze())))
        if show_test:
            plt.scatter(test_x[:, 0], test_x[:, 1], edgecolors='b',
                label='Test', facecolor=f.cmap(f.norm(test_y.squeeze())))
        plt.scatter(predictions[0][:, 0], predictions[0][:, 1], edgecolors='g',
                label=label, facecolor=f.cmap(f.norm(predictions[1].squeeze())))
        plt.legend()
        plt.title('Gaussian Input Dataset')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    @staticmethod
    def plot_datasets(data_handlers, inputs=None, predictions=None, labels=None,
                      show=True, filename=None, figsize=(10, 6)):
        """Plot several datasets of this class in one plot.

        Args:
            data_handlers: A list of GaussianData objects.
            inputs (optional): A list of numpy arrays representing inputs for
                each dataset.
            predictions (optional): A list of numpy arrays containing the
                predicted output values for the given input values.
            labels (optional): A label for each dataset.
            show: Whether the plot should be shown.
            filename (optional): If provided, the figure will be stored under
                this filename.
            figsize: A tuple, determining the size of the
                figure in inches.
        """
        n = len(data_handlers)
        assert((inputs is None and predictions is None) or \
               (inputs is not None and predictions is not None))
        assert((inputs is None or len(inputs) == n) and \
               (predictions is None or len(predictions) == n) and \
               (labels is None or len(labels) == n))

        fig, ax = plt.subplots(figsize=figsize)
        #plt.figure(figsize=figsize)
        plt.title('GaussianMixture tasks', size=20)

        # We need to produce a heatmap that spans all tasks.
        min_x = np.zeros((2, n))
        max_x = np.zeros((2, n))
        for i in range(n):
            data = data_handlers[i]

            train_x = data.get_train_inputs()
            test_x = data.get_test_inputs()
            mu = data._mean

            #dx = np.abs(np.vstack([train_x, test_x]) - mu[None, :]).max(axis=0)
            dx = max(np.abs(train_x - mu[None, :]).max(),
                     np.abs(test_x - mu[None, :]).max())

            min_x[:, i] = mu - dx
            max_x[:, i] = mu + dx

        min_x = min_x.min(axis=1)
        max_x = max_x.max(axis=1)

        slack = (max_x - min_x) * 0.02
        min_x -= slack
        max_x += slack

        grid_size = 1000
        x1 = np.linspace(start=min_x[0], stop=max_x[0], num=grid_size)
        x2 = np.linspace(start=min_x[1], stop=max_x[1], num=grid_size)

        X1, X2 = np.meshgrid(x1, x2)
        X = np.vstack([X1.ravel(), X2.ravel()]).T

        # Problem: Now that we have the underlying X mesh, how do we compute the
        # heatmap. Since every Gaussian has full support, we can't draw a
        # heatmap that displays all tasks with their correct Y value.
        # One options would be to just add all heat maps. For small variances
        # this would look "almost" correct.
        # Another option is to compute Voronoi cells for all tasks and compute
        # at each mesh position the y value corresponding to the task with the
        # nearest mean.

        # We decide here to compute y based on the nearest neighor, as this
        # seems to be the "most correct" plotting option.

        means = [d._mean for d in data_handlers]

        # Plot Voronoi diagram for debugging.
        #from scipy.spatial import Voronoi, voronoi_plot_2d
        #vor = Voronoi(means)
        #voronoi_plot_2d(vor)

        vor_tree = cKDTree(means)
        _, minds = vor_tree.query(X)

        Y = np.empty(X.shape[0])

        for i in range(n):
            mask = minds == i
            Y[mask] = data_handlers[i]._map(X[mask, :]).squeeze()

        Y = Y.reshape(X1.shape)

        f = plt.contourf(X1, X2, Y)
        plt.colorbar(f)

        colors = cm.rainbow(np.linspace(0, 1, n))

        phandlers = []
        plabels = []

        for i, data in enumerate(data_handlers):
            if labels is not None:
                lbl = labels[i]
            else:
                lbl = 'Predictions %d' % i

            if inputs is not None:
                p = plt.scatter(inputs[i][:, 0], inputs[i][:, 1],
                    edgecolors=colors[i],
                    facecolor=f.cmap(f.norm(predictions[i].squeeze())))
                phandlers.append(p)
                plabels.append(lbl)

        plt.legend(phandlers, plabels)
        plt.xlabel('x1')
        plt.ylabel('x2')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        if show:
            plt.show()

        return fig


if __name__ == '__main__':
    pass


