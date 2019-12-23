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
# @title           :regression1d_data.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :04/10/2019
# @version         :1.0
# @python_version  :3.6.8
"""
1D Regression Dataset
^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.special.regression1d_data` contains a data handler for a
CL toy regression problem. The user can construct individual datasets with this
data handler and use each of these datasets to train a model in a continual
leraning setting.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from warnings import warn

from utils import misc
from data.dataset import Dataset

class ToyRegression(Dataset):
    """An instance of this class shall represent a simple regression task.

    Attributes:
        train_x_range: The input range for training samples.
        test_x_range: The input range for test samples.
        val_x_range: The input range for validation samples.
    """
    def __init__(self, train_inter=[-10, 10], num_train=20,
                 test_inter=[-10, 10], num_test=80, val_inter=None,
                 num_val=None, map_function=lambda x : x, std=0, rseed=None):
        """Generate a new dataset.

        The input data x will be uniformly drawn for train samples and
        equidistant for test samples. The user has to specify a function that
        will map this random input data onto output samples y.

        Args:
            train_inter: A tuple, representing the interval from which x
                samples are drawn in the training set. Note, this range will
                apply to all input dimensions.
            num_train: Number of training samples.
            test_inter: A tuple, representing the interval from which x
                samples are drawn in the test set. Note, this range will
                apply to all input dimensions.
            num_test: Number of test samples.
            val_inter (optional): See parameter `test_inter`. If set, this
                argument leads to the construction of a validation set. Note,
                option `num_val` need to be specified as well.
            num_val (optional): Number of validation samples.
            map_function: A function handle that receives input
                samples and maps them to output samples.
            std: If not zero, Gaussian white noise with this std will be added
                to the training outputs.
            rseed: If ``None``, the current random state of numpy is used to
                   generate the data. Otherwise, a new random state with the
                   given seed is generated.
        """
        super().__init__()

        assert(val_inter is None and num_val is None or \
               val_inter is not None and num_val is not None)

        if rseed is None:
            rand = np.random
        else:
            rand = np.random.RandomState(rseed)

        train_x = rand.uniform(low=train_inter[0], high=train_inter[1],
                               size=(num_train, 1))
        test_x = np.linspace(start=test_inter[0], stop=test_inter[1],
                             num=num_test).reshape((num_test, 1))

        train_y = map_function(train_x)
        test_y = map_function(test_x)

        # Perturb training outputs.
        if std > 0:
            train_eps = rand.normal(loc=0.0, scale=std, size=(num_train, 1))
            train_y += train_eps

        # Create validation data if requested.
        if num_val is not None:
            val_x = np.linspace(start=val_inter[0], stop=val_inter[1],
                                num=num_val).reshape((num_val, 1))
            val_y = map_function(val_x)

            in_data = np.vstack([train_x, test_x, val_x])
            out_data = np.vstack([train_y, test_y, val_y])
        else:
            in_data = np.vstack([train_x, test_x])
            out_data = np.vstack([train_y, test_y])

        # Specify internal data structure.
        self._data['classification'] = False
        self._data['sequence'] = False
        self._data['in_data'] = in_data
        self._data['in_shape'] = [1]
        self._data['out_data'] = out_data
        self._data['out_shape'] = [1]
        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train, num_train + num_test)

        if num_val is not None:
            n_start = num_train + num_test
            self._data['val_inds'] = np.arange(n_start, n_start + num_val)

        self._map = map_function
        self._train_inter = train_inter
        self._test_inter = test_inter
        self._val_inter = val_inter

    @property
    def train_x_range(self):
        """Getter for read-only attribute :attr:`train_x_range`."""
        return self._train_inter

    @property
    def test_x_range(self):
        """Getter for read-only attribute :attr:`test_x_range`."""
        return self._test_inter

    @property
    def val_x_range(self):
        """Getter for read-only attribute :attr:`val_x_range`."""
        return self._val_inter

    def _get_function_vals(self, num_samples=100, x_range=None):
        """Get real function values for equidistant x values in a range that
        covers the test and training data. These values can be used to plot the
        ground truth function.

        Args:
            num_samples: Number of samples to be produced.
            x_range: If a specific range should be used to gather function
                values.

        Returns:
            x, y: Two numpy arrays containing the corresponding x and y values.
        """
        if x_range is None:
            min_x = min(self._train_inter[0], self._test_inter[0])
            max_x = max(self._train_inter[1], self._test_inter[1])
            if self.num_val_samples > 0:
                min_x = min(min_x, self._val_inter[0])
                max_x = max(max_x, self._val_inter[1])
        else:
            min_x = x_range[0]
            max_x = x_range[1]

        slack_x = 0.05 * (max_x - min_x)

        sample_x = np.linspace(start=min_x-slack_x, stop=max_x+slack_x,
                               num=num_samples).reshape((num_samples, 1))
        sample_y = self._map(sample_x)

        return sample_x, sample_y

    def plot_dataset(self, show=True):
        """Plot the whole dataset.

        Args:
            show: Whether the plot should be shown.
        """

        train_x = self.get_train_inputs().squeeze()
        train_y = self.get_train_outputs().squeeze()

        test_x = self.get_test_inputs().squeeze()
        test_y = self.get_test_outputs().squeeze()

        if self.num_val_samples > 0:
            val_x = self.get_val_inputs().squeeze()
            val_y = self.get_val_outputs().squeeze()

        sample_x, sample_y = self._get_function_vals()

        # The default matplotlib setting is usually too high for most plots.
        plt.locator_params(axis='y', nbins=2)
        plt.locator_params(axis='x', nbins=6)

        plt.plot(sample_x, sample_y, color='k', label='f(x)',
                 linestyle='dashed', linewidth=.5)
        plt.scatter(train_x, train_y, color='r', label='Train')
        plt.scatter(test_x, test_y, color='b', label='Test', alpha=0.8)
        if self.num_val_samples > 0:
            plt.scatter(val_x, val_y, color='g', label='Val', alpha=0.5)
        plt.legend()
        plt.title('1D-Regression Dataset')
        plt.xlabel('$x$')
        plt.ylabel('$y$')

        if show:
            plt.show()

    def plot_predictions(self, predictions, label='Pred', show_train=True,
                         show_test=True):
        """Plot the dataset as well as predictions.

        Args:
            predictions: A tuple of x and y values, where the y values are
                         computed by a trained regression network.
                         Note, that we assume the x values to be sorted.
            label: Label of the predicted values as shown in the legend.
            show_train: Show train samples.
            show_test: Show test samples.
        """
        train_x = self.get_train_inputs().squeeze()
        train_y = self.get_train_outputs().squeeze()
        
        test_x = self.get_test_inputs().squeeze()
        test_y = self.get_test_outputs().squeeze()

        sample_x, sample_y = self._get_function_vals()
        plt.plot(sample_x, sample_y, color='k', label='f(x)',
                 linestyle='dashed', linewidth=.5)
        if show_train:
            plt.scatter(train_x, train_y, color='r', label='Train')
        if show_test:
            plt.scatter(test_x, test_y, color='b', label='Test')
        plt.scatter(predictions[0], predictions[1], color='g', label=label)
        plt.legend()
        plt.title('1D-Regression Dataset')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.show()

    def get_identifier(self):
        """Returns the name of the dataset."""
        return '1DRegression'

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
        assert( outputs is not None or predictions is not None)

        plt.figure(figsize=figsize)
        plt.title(title, size=20)
        if interactive:
            plt.ion()

        sample_x, sample_y = self._get_function_vals()
        plt.plot(sample_x, sample_y, color='k', label='f(x)',
                 linestyle='dashed', linewidth=.5)
        if outputs is not None:
            plt.scatter(inputs, outputs, color='b', label='Targets')
        if predictions is not None:
            plt.scatter(inputs, predictions, color='r', label='Predictions')
        plt.legend()
        plt.xlabel('$x$')
        plt.ylabel('$y$')

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

    @staticmethod
    def plot_datasets(data_handlers, inputs=None, predictions=None, labels=None,
                      fun_xranges=None, show=True, filename=None,
                      figsize=(10, 6), publication_style=False):
        """Plot several datasets of this class in one plot.

        Args:
            data_handlers: A list of ToyRegression objects.
            inputs (optional): A list of numpy arrays representing inputs for
                each dataset.
            predictions (optional): A list of numpy arrays containing the
                predicted output values for the given input values.
            labels (optional): A label for each dataset.
            fun_xranges (optional): List of x ranges in which the true
                underlying function per dataset should be sketched.
            show: Whether the plot should be shown.
            filename (optional): If provided, the figure will be stored under
                this filename.
            figsize: A tuple, determining the size of the figure in inches.
            publication_style: Whether the plots should be in publication style.
        """
        n = len(data_handlers)
        assert((inputs is None and predictions is None) or \
               (inputs is not None and predictions is not None))
        assert((inputs is None or len(inputs) == n) and \
               (predictions is None or len(predictions) == n) and \
               (labels is None or len(labels) == n))
        assert(fun_xranges is None or len(fun_xranges) == n)

        # Set-up matplotlib to adhere to our graphical conventions.
        #misc.configure_matplotlib_params(fig_size=1.2*np.array([1.6, 1]),
        #                                 font_size=8)

        # Get a colorscheme from colorbrewer2.org.
        colors = misc.get_colorbrewer2_colors(family='Dark2')
        if n > len(colors):
            warn('Changing to automatic color scheme as we don\'t have ' +
                 'as many manual colors as tasks.')
            colors = cm.rainbow(np.linspace(0, 1, n))

        if publication_style:
            ts, lw, ms = 60, 15, 140 # text fontsize, line width, marker size
            figsize = (12, 6)
        else:
            ts, lw, ms = 12, 2, 15

        fig, axes = plt.subplots(figsize=figsize)
        plt.title('1D regression', size=ts, pad=ts)

        phandlers = []
        plabels = []

        for i, data in enumerate(data_handlers):
            if labels is not None:
                lbl = labels[i]
            else:
                lbl = 'Function %d' % i

            fun_xrange = None
            if fun_xranges is not None:
                fun_xrange = fun_xranges[i]
            sample_x, sample_y = data._get_function_vals(x_range=fun_xrange)
            p, = plt.plot(sample_x, sample_y, color=colors[i],
                          linestyle='dashed', linewidth=lw/3)

            phandlers.append(p)
            plabels.append(lbl)
            if inputs is not None:
                p = plt.scatter(inputs[i], predictions[i], color=colors[i],
                    s=ms)
                phandlers.append(p)
                plabels.append('Predictions')

        if publication_style:
            axes.grid(False)
            axes.set_facecolor('w')
            axes.axhline(y=axes.get_ylim()[0], color='k', lw=lw)
            axes.axvline(x=axes.get_xlim()[0], color='k', lw=lw)
            if len(data_handlers)==3:
                plt.yticks([-1, 0, 1], fontsize=ts)
                plt.xticks([-2.5, 0, 2.5], fontsize=ts)
            else:
                for tick in axes.yaxis.get_major_ticks():
                    tick.label.set_fontsize(ts) 
                for tick in axes.xaxis.get_major_ticks():
                    tick.label.set_fontsize(ts) 
            axes.tick_params(axis='both', length=lw, direction='out', width=lw/2.)
        else:
            plt.legend(phandlers, plabels)

        plt.xlabel('$x$', fontsize=ts)
        plt.ylabel('$y$', fontsize=ts)
        plt.tight_layout()

        if filename is not None:
            #plt.savefig(filename + '.pdf', bbox_inches='tight')
            plt.savefig(filename, bbox_inches='tight')

        if show:
            plt.show()

if __name__ == '__main__':
    pass


