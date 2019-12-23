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
# @title           :cifar10_data.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :08/08/2018
# @version         :1.0
# @python_version  :3.6.6
"""
CIFAR-10 Dataset
----------------

The module :mod:`data.cifar10_data` contains a handler for the CIFAR 10 dataset.

The dataset consists of 60000 32x32 colour images in 10 classes, with 6000
images per class. There are 50000 training images and 10000 test images.

Information about the dataset can be retrieved from:
    https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import numpy as np
import time
import _pickle as pickle
import urllib.request
import tarfile
import matplotlib.pyplot as plt
from warnings import warn

from data.dataset import Dataset

class CIFAR10Data(Dataset):
    """An instance of the class shall represent the CIFAR-10 dataset.

    Note, the constructor does not safe a data dump (via pickle) as, for
    instance, the MNIST data handler (:class:`data.mnist_data.MNISTData`) does.
    The reason is, that the downloaded files are already in a nice to read
    format, such that the time saved to read the file from a dump file is
    minimal.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be represented in a
            one-hot encoding.
        use_data_augmentation (bool, optional): Note, this option currently only
            applies to input batches that are transformed using the class
            member :meth:`data.dataset.Dataset.input_to_torch_tensor` (hence,
            **only available for PyTorch**, so far).
        validation_size (int): The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
    """
    _DOWNLOAD_PATH = 'https://www.cs.toronto.edu/~kriz/'
    _DOWNLOAD_FILE = 'cifar-10-python.tar.gz'
    _EXTRACTED_FOLDER = 'cifar-10-batches-py'

    _TRAIN_BATCH_FNS = ['data_batch_%d' % i for i in range(1,6)]
    _TEST_BATCH_FN = 'test_batch'
    _META_DATA_FN = 'batches.meta'

    # In which file do we dump the dataset, to allow a faster readout next
    # time?
    _CIFAR10_DATA_DUMP = 'cifar10_dataset.pickle'

    def __init__(self, data_path, use_one_hot=False,
                 use_data_augmentation=False, validation_size=5000):
        super().__init__()

        start = time.time()

        print('Reading CIFAR-10 dataset ...')

        if not os.path.exists(data_path):
            print('Creating directory "%s" ...' % (data_path))
            os.makedirs(data_path)

        extracted_data_dir = os.path.join(data_path, 
                                          CIFAR10Data._EXTRACTED_FOLDER)

        # If data has been processed before.
        build_from_scratch = True
        dump_fn = os.path.join(extracted_data_dir,
                               CIFAR10Data._CIFAR10_DATA_DUMP)
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

                if self.num_val_samples != validation_size:
                    build_from_scratch = True


        if build_from_scratch:
            archive_fn = os.path.join(data_path, CIFAR10Data._DOWNLOAD_FILE)


            if not os.path.exists(extracted_data_dir):
                print('Downloading dataset ...')
                urllib.request.urlretrieve(CIFAR10Data._DOWNLOAD_PATH + \
                                           CIFAR10Data._DOWNLOAD_FILE, \
                                           archive_fn)

                # Extract downloaded dataset.
                tar = tarfile.open(archive_fn, "r:gz")
                tar.extractall(path=data_path)
                tar.close()

                os.remove(archive_fn)

            train_batch_fns = list(map(lambda p : os.path.join(
                extracted_data_dir, p), CIFAR10Data._TRAIN_BATCH_FNS))
            test_batch_fn = os.path.join(extracted_data_dir, 
                                         CIFAR10Data._TEST_BATCH_FN)
            meta_fn = os.path.join(extracted_data_dir, 
                                   CIFAR10Data._META_DATA_FN)

            assert(all(map(os.path.exists, train_batch_fns)) and
                   os.path.exists(test_batch_fn) and os.path.exists(meta_fn))

            self._data['classification'] = True
            self._data['sequence'] = False
            self._data['num_classes'] = 10
            self._data['is_one_hot'] = use_one_hot
            
            self._data['in_shape'] = [32, 32, 3]
            self._data['out_shape'] = [10 if use_one_hot else 1]

            # Fill the remaining _data fields with the information read from
            # the downloaded files.
            self._read_meta(meta_fn)
            self._read_batches(train_batch_fns, test_batch_fn, validation_size)

            # As the time advantage are minimal compared to the huge storage
            # requirements, we don't safe the data as pickle file.
            ## Save read dataset to allow faster reading in future.
            #with open(dump_fn, 'wb') as f:
            #    pickle.dump(self._data, f)

        # Initialize PyTorch data augmentation.
        self._augment_inputs = False
        if use_data_augmentation:
            self._augment_inputs = True
            self._train_transform, self._test_transform = \
                CIFAR10Data.torch_input_transforms()

        end = time.time()
        print('Elapsed time to read dataset: %f sec' % (end-start))

    def _read_meta(self, filename):
        """Read the meta data file.

        This method will add an additional field to the ``_data`` attribute
        named "cifar10". This dictionary will be filled with two members:

            - "label_names": The names of the associated categorical class
                labels.
            - "num_cases_per_batch": The number of samples in each batch.

        Args:
            filename: The path to the meta data file.
        """
        with open(filename, 'rb') as f:
            meta_data = pickle.load(f, encoding='UTF-8')

        assert(meta_data['num_vis'] == 32 * 32 * 3)

        self._data['cifar10'] = dict()

        self._data['cifar10']['label_names'] = meta_data['label_names']
        self._data['cifar10']['num_cases_per_batch'] = \
            meta_data['num_cases_per_batch']

    def _read_batches(self, train_fns, test_fn, validation_size):
        """Read all batches from files.

        The method fills the remaining mandatory fields of the _data attribute,
        that have not been set yet in the constructor.

        The images are converted to match the output shape (32, 32, 3) and
        scaled to have values between 0 and 1. For labels, the correct encoding
        is enforced.

        Args:
            train_fns: The filepaths of the different training batches (files
                are assumed to be in order).
            test_fn: Filepath of the test batch.
            validation_size: Number of validation samples.
        """
        with open(test_fn, 'rb') as f:
            test_batch = pickle.load(f, encoding='bytes')

        # Note, that we ignore the two keys: "batch_label" and "filenames".
        test_labels = np.array(test_batch['labels'.encode()])
        test_samples = test_batch['data'.encode()]
        
        # Read training batches.
        for i, fn in enumerate(train_fns):
            with open(fn, 'rb') as f:
                curr_batch = pickle.load(f, encoding='bytes')

            curr_labels = np.array(curr_batch['labels'.encode()])
            curr_samples = curr_batch['data'.encode()]

            if i == 0:
                train_labels = curr_labels
                train_samples = curr_samples
            else:
                train_labels = np.concatenate((train_labels, curr_labels))
                train_samples = np.concatenate((train_samples, curr_samples),
                                               axis=0)

        if validation_size > 0:
            assert(validation_size < train_labels.shape[0])
            val_inds = np.arange(validation_size)
            train_inds = np.arange(validation_size, train_labels.size)

        else:
            train_inds = np.arange(train_labels.size)

        test_inds = np.arange(train_labels.size, 
                              train_labels.size + test_labels.size)

        labels = np.concatenate([train_labels, test_labels])
        labels = np.reshape(labels, (-1, 1))

        images = np.concatenate([train_samples, test_samples], axis=0)

        # Note, images are currently encoded in a way, that there shape
        # corresponds to (3, 32, 32). For consistency reasons, we would like to
        # change that to (32, 32, 3).
        images = np.reshape(images, (-1, 3, 32, 32))
        images = np.rollaxis(images, 1, 4)
        images = np.reshape(images, (-1, 32 * 32 * 3))
        # Scale images into a range between 0 and 1.
        images = images / 255

        self._data['in_data'] = images
        self._data['train_inds'] = train_inds
        self._data['test_inds'] = test_inds
        if validation_size > 0:
            self._data['val_inds'] = val_inds

        if self._data['is_one_hot']:
            labels = self._to_one_hot(labels)

        self._data['out_data'] = labels

    def _get_batch_identifier(self, index):
        """Return the identifier of the batch a given sample is drawn from.

        Batches 1 to 5 are the training batches. Batch 6 is the test batch.

        Args:
            index: The sample index (row index) in _data['in_data'].
        """
        return index % 10000 + 1

    def plot_sample(self, image, label=None, figsize = 1.5, interactive=False,
                    file_name=None):
        """Plot a single CIFAR-10 sample.

        This method is thought to be helpful for evaluation and debugging
        purposes.

        .. deprecated:: 1.0
            Please use method :meth:`data.dataset.Dataset.plot_samples` instead.

        Args:
            image: A single CIFAR-10 image (given as 1D vector).
            label: The label of the given image.
            figsize: The height and width of the displayed image.
            interactive: Turn on interactive mode. Thus program will run in
                background while figure is displayed. The figure will be
                displayed until another one is displayed, the user closes it or
                the program has terminated. If this option is deactivated, the
                program will freeze until the user closes the figure.
            file_name: (optional) If a file name is provided, then the image
                will be written into a file instead of plotted to the screen.
        """
        warn('Please use method "plot_samples" instead.', DeprecationWarning)

        plt.figure(figsize = (figsize, figsize))

        if label is None:
            plt.title("CIFAR-10 Sample")
        else:
            label_name = self._data['cifar10']['label_names'][label]
            plt.title('Label of shown sample: %s (%d)' % (label_name, label))
        plt.axis('off')
        if interactive:
            plt.ion()
        plt.imshow(np.reshape(image, self.in_shape))
        if file_name is not None:
            plt.savefig(file_name, bbox_inches='tight')
        else:
            plt.show()

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'CIFAR-10'

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        Note, this method has been overwritten from the base class.

        The input images are preprocessed if data augmentation is enabled.
        Preprocessing involves normalization and (for training mode) random
        perturbations.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor.
        """
        if self._augment_inputs and not force_no_preprocessing:
            if mode == 'inference':
                transform = self._test_transform
            elif mode == 'train':
                transform = self._train_transform
            else:
                raise ValueError('"%s" not a valid value for argument "mode".'
                                 % mode)

            return CIFAR10Data.torch_augment_images(x, device, transform)

        else:
            return Dataset.input_to_torch_tensor(self, x, device,
                mode=mode, force_no_preprocessing=force_no_preprocessing)

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("CIFAR-10 Sample")
        else:
            assert(np.size(outputs) == 1)
            label = np.asscalar(outputs)
            label_name = self._data['cifar10']['label_names'][label]

            if predictions is None:
                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label))
            else:
                if np.size(predictions) == self.num_classes:
                    pred_label = np.argmax(predictions)
                else:
                    pred_label = np.asscalar(predictions)
                pred_label_name = \
                    self._data['cifar10']['label_names'][pred_label]

                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label) + '\nPrediction: %s (%d)' % \
                             (pred_label_name, pred_label))

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

    @staticmethod
    def torch_input_transforms():
        """Get data augmentation pipelines for CIFAR-10 inputs.

        Note, the augmentation is inspired by the augmentation proposed in:
            https://www.aiworkbox.com/lessons/augment-the-cifar10-dataset-using\
-the-randomhorizontalflip-and-randomcrop-transforms

        Note:
            We use the same data augmentation pipeline for CIFAR-100, as the
            images are very similar. Here is an example where they use slightly
            different normalization values, but we ignore this for now:
            https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html

        Returns:
            (tuple): Tuple containing:

                - **train_transform**: A transforms pipeline that applies random
                  transformations and normalizes the image.
                - **test_transform**: Similar to train_transform, but no random
                  transformations are applied.
        """
        # Copyright 2017-2018 aiworkbox.com
        # Unfortunately, no license was visibly provided with this code.
        # Though, we note that the original license applies regarding the parts
        # of the code that have been copied from the above mentioned website (we
        # slightly modified this code).
        #
        # Note, that we use this code WITHOUT ANY WARRANTIES.

        import torchvision.transforms as transforms

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.ToPILImage('RGB'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=[32,32], padding=4),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage('RGB'),
            transforms.ToTensor(),
            normalize,
        ])

        return train_transform, test_transform

    @staticmethod
    def torch_augment_images(x, device, transform):
        """Augment CIFAR-10 images using a given PyTorch transformation.

        Args:
            x (numpy.ndarray): A 2D-Numpy array containing CIFAR-10 images.
            device (torch.device or int): The PyTorch device on which the
                resulting tensor should be.
            transform: A :mod:`torchvision.transforms` method to modify the
                data.

        Returns:
            (torch.Tensor): The augmented images as PyTorch tensor.
        """
        from torch import stack

        #x = torch.from_numpy(x).float()
        #x = x.view([-1, 32, 32, 3])
        #x = x.permute(0, 3, 1, 2)
        #x = torch.stack([transform(_x) for _x in x])
        #return x.to(device)

        assert(len(x.shape) == 2) # batch size plus flattened image.

        # Transform the numpy data into a representation as expected by the
        # ToPILImage transformation.
        x = (x * 255.0).astype('uint8')
        x = x.reshape(-1, 32, 32, 3)

        x = stack([transform(x[i,...]) for i in range(x.shape[0])]).to(device)

        # Transform tensor back to numpy shape.
        # FIXME This is a horrible solution, but at least we ensure that the
        # user gets a tensor in the same shape as always and does not have to
        # deal with cases.
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, 3072) # 3072 = 32 * 32 * 3

        return x

if __name__ == '__main__':
    pass


