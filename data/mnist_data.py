#!/usr/bin/env python3
# Copyright 2018 Christian Henning
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
# @title           :mnist_data.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :08/08/2018
# @version         :1.0
# @python_version  :3.6.6
"""
MNIST Dataset
-------------

The module :mod:`data.mnist_data` contains a handler for the MNIST dataset.

The implementation is based on an earlier implementation of a class I used in
another project:

    https://git.io/fNyQL

Information about the dataset can be retrieved from:

    http://yann.lecun.com/exdb/mnist/
"""

import os
import struct
import numpy as np
import time
import _pickle as pickle
import urllib.request
import gzip
import matplotlib.pyplot as plt
from warnings import warn

from data.dataset import Dataset

class MNISTData(Dataset):
    """An instance of the class shall represent the MNIST dataset.

    The constructor checks whether the dataset has been read before (a pickle
    dump has been generated). If so, it reads the dump. Otherwise, it
    reads the data from scratch and creates a dump for future usage.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be
            represented in a one-hot encoding.
        validation_size (int): The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
    """
    _DOWNLOAD_PATH = 'http://yann.lecun.com/exdb/mnist/'
    _TRAIN_IMGS_FN = 'train-images-idx3-ubyte.gz'
    _TRAIN_LBLS_FN = 'train-labels-idx1-ubyte.gz'
    _TEST_IMGS_FN = 't10k-images-idx3-ubyte.gz'
    _TEST_LBLS_FN = 't10k-labels-idx1-ubyte.gz'
    # In which file do we dump the dataset, to allow a faster readout next
    # time?
    _MNIST_DATA_DUMP = 'mnist_dataset.pickle'
    # In which subfolder of the datapath should the data be stored.
    _SUBFOLDER = 'MNIST'
    
    def __init__(self, data_path, use_one_hot=False, validation_size=5000):
        super().__init__()

        start = time.time()

        print('Reading MNIST dataset ...')
        
        # Actual data path
        data_path = os.path.join(data_path, MNISTData._SUBFOLDER)

        if not os.path.exists(data_path):
            print('Creating directory "%s" ...' % (data_path))
            os.makedirs(data_path)

        # If data has been processed before.
        build_from_scratch = True
        dump_fn = os.path.join(data_path, MNISTData._MNIST_DATA_DUMP)
        if os.path.isfile(dump_fn):
            build_from_scratch = False

            with open(dump_fn, 'rb') as f:
                self._data = pickle.load(f)

                if self._data['is_one_hot'] != use_one_hot:
                    reverse = True
                    if use_one_hot:
                        reverse = False

                    self._data['is_one_hot'] = use_one_hot
                    self._data['out_data'] = self._to_one_hot(
                            self._data['out_data'], reverse=reverse)
                    self._data['out_shape'] = [self._data['out_data'].shape[1]]

                # DELETEME A previous version of the dataloader stored the
                # validation set in the pickle file. Hence, this line ensures
                # downwards compatibility.
                if self.num_val_samples != 0:
                    build_from_scratch = True
                    self._data['val_inds'] = None


        if build_from_scratch:
            train_images_fn = os.path.join(data_path, MNISTData._TRAIN_IMGS_FN)
            train_labels_fn = os.path.join(data_path, MNISTData._TRAIN_LBLS_FN)
            test_images_fn = os.path.join(data_path, MNISTData._TEST_IMGS_FN)
            test_labels_fn = os.path.join(data_path, MNISTData._TEST_LBLS_FN)

            if not os.path.exists(train_images_fn):
                print('Downloading training images ...')
                urllib.request.urlretrieve(MNISTData._DOWNLOAD_PATH + \
                                           MNISTData._TRAIN_IMGS_FN, \
                                           train_images_fn)

                ## Extract downloaded images.
                #with gzip.open(train_images_fn, 'rb') as f_in:
                #     with open(os.path.splitext(train_images_fn)[0], \
                #               'wb') as f_out:
                #         shutil.copyfileobj(f_in, f_out)

            if not os.path.exists(train_labels_fn):
                print('Downloading training labels ...')
                urllib.request.urlretrieve(MNISTData._DOWNLOAD_PATH + \
                                           MNISTData._TRAIN_LBLS_FN, \
                                           train_labels_fn)

            if not os.path.exists(test_images_fn):
                print('Downloading test images ...')
                urllib.request.urlretrieve(MNISTData._DOWNLOAD_PATH + \
                                           MNISTData._TEST_IMGS_FN, \
                                           test_images_fn)

            if not os.path.exists(test_labels_fn):
                print('Downloading test labels ...')
                urllib.request.urlretrieve(MNISTData._DOWNLOAD_PATH + \
                                           MNISTData._TEST_LBLS_FN, \
                                           test_labels_fn)

            # read labels
            train_labels = MNISTData._read_labels(train_labels_fn)
            test_labels = MNISTData._read_labels(test_labels_fn)

            # read images
            train_inputs = MNISTData._read_images(train_images_fn)
            test_inputs = MNISTData._read_images(test_images_fn)
            
            assert(train_labels.shape[0] == train_inputs.shape[0])
            assert(test_labels.shape[0] == test_inputs.shape[0])

            # Note, we ignore a possible validation set here on purpose, as it
            # should not be part of the pickle (see below).
            train_inds = np.arange(train_labels.size)
            test_inds = np.arange(train_labels.size, 
                                  train_labels.size + test_labels.size)

            labels = np.concatenate([train_labels, test_labels])
            images = np.concatenate([train_inputs, test_inputs], axis=0)

            labels = np.reshape(labels, (-1, 1))
            # Scale images into a range between 0 and 1.
            images = images / 255

            # Bring these raw readings into the internal structure of the
            # Dataset class
            self._data['classification'] = True
            self._data['sequence'] = False
            self._data['num_classes'] = 10
            self._data['is_one_hot'] = use_one_hot
            self._data['in_data'] = images
            self._data['in_shape'] = [28, 28, 1]
            self._data['out_shape'] = [10 if use_one_hot else 1]
            self._data['train_inds'] = train_inds
            self._data['test_inds'] = test_inds

            if use_one_hot:
                labels = self._to_one_hot(labels)

            self._data['out_data'] = labels

            # Save read dataset to allow faster reading in future.
            with open(dump_fn, 'wb') as f:
                pickle.dump(self._data, f)

        # After writing the pickle, correct train and validation set indices.
        if validation_size > 0:
            train_inds_orig = self._data['train_inds']
            assert(validation_size < train_inds_orig.size)

            val_inds = np.arange(validation_size)
            train_inds = np.arange(validation_size, train_inds_orig.size)

            self._data['train_inds'] = train_inds
            self._data['val_inds'] = val_inds

        end = time.time()
        print('Elapsed time to read dataset: %f sec' % (end-start))

    @staticmethod
    def _read_labels(filename):
        """Reading a set of labels from a file.

        Args:
            filename: Path and name of the byte file that contains the labels.

        Returns:
            The labels as a 1D numpy array.
        """
        assert(os.path.isfile(filename))

        print('Reading labels from %s.' % filename)
        with gzip.open(filename, "rb") as f:
            # Skip magic number.
            f.read(4)
            # Get number of labels in this file.
            num = int.from_bytes(f.read(4), byteorder='big')
            print('Number of labels in current file: %d' % num)

            # The rest of the file are "num" bytes, each byte encoding a label.
            labels = np.array(struct.unpack('%dB' % num, f.read(num)))

            return labels

    @staticmethod
    def _read_images(filename):
        """Reading a set of images from a file.

        Args:
            filename: Path and name of the byte file that contains the images.

        Returns:
            The images stacked in a 2D array, where each row is one image.
        """
        assert(os.path.isfile(filename))

        print('Reading images from %s.' % filename)
        with gzip.open(filename, 'rb') as f:
            # Skip magic number
            f.read(4)
            # Get number of images in this file.
            num = int.from_bytes(f.read(4), byteorder='big')
            print('Number of images in current file: %d' % num)
            # Get number of rows and columns.
            rows = int.from_bytes(f.read(4), byteorder='big')
            cols = int.from_bytes(f.read(4), byteorder='big')

            # The rest of the file consists of pure image data, each pixel
            # value encoded as a byte.
            num_rem_bytes = num * rows * cols
            images = np.array(struct.unpack('%dB' % num_rem_bytes,
                                            f.read(num_rem_bytes)))

            images = np.reshape(images, (-1, rows * cols))

            return images

    @staticmethod
    def plot_sample(image, label=None, interactive=False, file_name=None):
        """Plot a single MNIST sample.

        This method is thought to be helpful for evaluation and debugging
        purposes.

        .. deprecated:: 1.0
            Please use method :meth:`data.dataset.Dataset.plot_samples` instead.

        Args:
            image: A single MNIST image (given as 1D vector).
            label: The label of the given image.
            interactive: Turn on interactive mode. Thus program will run in
                background while figure is displayed. The figure will be
                displayed until another one is displayed, the user closes it or
                the program has terminated. If this option is deactivated, the
                program will freeze until the user closes the figure.
            file_name: (optional) If a file name is provided, then the image
                will be written into a file instead of plotted to the screen.
        """
        warn('Please use method "plot_samples" instead.', DeprecationWarning)

        if label is None:
            plt.title("MNIST Sample")
        else:
            plt.title('Label of shown sample: %d' % label)
        plt.axis('off')
        if interactive:
            plt.ion()
        plt.imshow(np.reshape(image, (28, 28)))
        if file_name is not None:
            plt.savefig(file_name, bbox_inches='tight')
        else:
            plt.show()

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'MNIST'

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("MNIST Sample")
        else:
            assert(np.size(outputs) == 1)
            label = np.asscalar(outputs)
            
            if predictions is None:
                ax.set_title('MNIST sample with\nlabel: %d' % label)
            else:
                if np.size(predictions) == self.num_classes:
                    pred_label = np.argmax(predictions)
                else:
                    pred_label = np.asscalar(predictions)
                    
                ax.set_title('MNIST sample with\nlabel: %d (prediction: %d)' %
                             (label, pred_label))

        #plt.subplots_adjust(wspace=0.5, hspace=0.4)

        ax.set_axis_off()
        ax.imshow(np.squeeze(np.reshape(inputs, self.in_shape)))
        fig.add_subplot(ax)

        if num_inner_plots == 2:
            ax = plt.Subplot(fig, inner_grid[1])
            ax.set_title('Predictions')
            bars = ax.bar(range(self.num_classes), np.squeeze(predictions))
            ax.set_xticks(range(self.num_classes))
            if outputs is not None:
                bars[int(label)].set_color('r')
            fig.add_subplot(ax)

    def _plot_config(self, inputs, outputs=None, predictions=None):
        """Re-Implementation of method
        :meth:`data.dataset.Dataset._plot_config`.

        This method has been overriden to ensure, that there are 2 subplots,
        in case the predictions are given.
        """
        plot_configs = super()._plot_config(inputs, outputs=outputs,
                                            predictions=predictions)
        
        if predictions is not None and \
                np.shape(predictions)[1] == self.num_classes:
            plot_configs['outer_hspace'] = 0.6
            plot_configs['inner_hspace'] = 0.4
            plot_configs['num_inner_rows'] = 2
            #plot_configs['num_inner_cols'] = 1
            plot_configs['num_inner_plots'] = 2

        return plot_configs

if __name__ == '__main__':
    pass


