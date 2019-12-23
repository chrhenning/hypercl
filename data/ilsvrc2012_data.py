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
# @title           :data/ilsvrc2012_data.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :05/13/2019
# @version         :1.0
# @python_version  :3.6.8
"""
ILSVRC2012 Dataset
------------------

The module :mod:`data.ilsvrc2012_data` contains a  handler for the Imagenet
Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) dataset, a subset of
the ImageNet dataset:

    http://www.image-net.org/challenges/LSVRC/2012/index

For more details on the dataset, please refer to:

    Olga Russakovsky et al. ImageNet Large Scale Visual Recognition Challenge.
    *International Journal of Computer Vision 115*, no. 3 (December 1, 2015):
    211–52, https://doi.org/10.1007/s11263-015-0816-y

.. note::
    In the current implementation, this handler will not download and extract
    the dataset for you. You have to do this manually by following the
    instructions of the README file (which is located in the same folder as this
    file).

.. note::
    We use the validation set as test set. A new (custom) validation set will
    be created by taking the first :math:`n` samples from each training class as
    validation samples, where :math:`n` is configured by the user.

.. note::
    This dataset has not yet been prepared for Tensorflow use!

When using PyTorch, this class will create dataset classes
(:class:`torch.utils.data.Dataset`) for you for the training, testing and
validation set. Afterwards, you can use these dataset instances to create data
loaders:

.. code-block:: python

    train_loader = torch.utils.data.DataLoader(
        ilsvrc2012_data.torch_train, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True)

You should then use these Pytorch data loaders rather than class internal
methods to work with the dataset.

PyTorch data augmentation is applied as defined by the method
:meth:`ILSVRC2012Data.torch_input_transforms`. Images will be resized and
cropped to size 224 x 224.
"""
# FIXME We currently rely too much on the internals of class ImageFolder.
import torchvision
import warnings
from packaging import version
if version.parse(torchvision.__version__) < version.parse('0.2.2'):
    # FIXME Probably not necessary to enforce, just ignore non-existing
    # "targets" field.
    raise Exception('Code requires torchvision to have at least version ' +
                    '"0.2.2" (current version: %s).' % torchvision.__version__)
elif version.parse(torchvision.__version__) != version.parse('0.2.2'):
    warnings.warn('Code not been tested with torchvision version %s!'
                  % torchvision.__version__)

import os
import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings

from data.large_img_dataset import LargeImgDataset

class ILSVRC2012Data(LargeImgDataset):
    """An instance of the class shall represent the ILSVRC2012 dataset.

    The input data of the dataset will be strings to image files. The output
    data corresponds to object labels according to the ``ILSVRC2012_ID`` - 1.

    Note:
        This is different from many other ILSVRC2012 data handlers, where the
        labels are computed based on the order of the training folder names
        (which correspond to WordNet IDs (``WNID``)).

    Note:
        The dataset has to be already downloaded and extracted before
        this method can be called. See the local README file for details.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be
            represented in a one-hot encoding. Note, class labels
            correspond to the ``ILSVRC2012_ID`` minus 1 (from 0 to 999).

            .. note::
                This option does not influence the internal PyTorch
                Dataset classes (e.g., cmp.
                :attr:`data.large_img_dataset.LargeImgDataset.torch_train`),
                that can be used in conjunction with PyTorch data loaders.
        num_val_per_class (int): The number of validation samples per class.

            .. note::
                The actual ILSVRC2012 validation set is used as test set
                by this data handler. Therefore, a new validation set is
                constructed (if value greater than 0), using the same amount of
                samples per class.
                For instance: If value 50 is given, a validation set of size
                50 * 1000 = 50,000 is constructed (these samples will be removed
                from the training set).

            .. note::
                Validation samples use the same data augmentation pipeline
                as test samples.
    """
    _TRAIN_FOLDER = 'train'
    _VAL_FOLDER = 'val'
    _META_FILE = os.path.join('meta', 'data', 'meta.mat')

    def __init__(self, data_path, use_one_hot=False, num_val_per_class=0):
        # 732 is the minimum number of training samples per class in
        # ILSVRC2012.
        assert(num_val_per_class < 732)
        # We keep the full path to each image in memory, so we don't need to
        # tell the super class the root path to each image (i.e., samples
        # contain absolute not relative paths).
        super().__init__('')

        start = time.time()

        print('Reading ILSVRC2012 dataset ...')

        meta_fn = os.path.join(data_path, ILSVRC2012Data._META_FILE)
        train_dir = os.path.join(data_path, ILSVRC2012Data._TRAIN_FOLDER)
        val_dir = os.path.join(data_path, ILSVRC2012Data._VAL_FOLDER)

        err_msg = 'Please follow the steps described in the file ' + \
            'data/README.md to download and extract the data.'
        if not os.path.exists(train_dir):
            raise FileNotFoundError('Training images not found in directory ' +
                train_dir + '.\n' + err_msg)
        elif not os.path.exists(val_dir):
            raise FileNotFoundError('Validation images not found in ' +
                'directory ' + val_dir + '.\n' + err_msg)
        elif not os.path.exists(meta_fn):
            raise FileNotFoundError('Meta file not found: ' +
                meta_fn + '.\n' + err_msg)

        # Read meta file.
        self._data['meta'] = dict()
        self._read_meta(meta_fn)

        # Read dataset.
        self._process_dataset(train_dir, val_dir, use_one_hot,
                              num_val_per_class)

        # Translate everything into the internal structure of this class.
        num_train = len(self._torch_ds_train)
        num_test = len(self._torch_ds_test)
        num_val = 0 if self._torch_ds_val is None else \
            len(self._torch_ds_val)
        num_samples = num_train + num_test + num_val
        # Just a sanity check, as these numbers should be fixed whenever the
        # full dataset is loaded.
        if num_test != 50000:
            warnings.warn('ILSVRC2012 should contain 50,000 test samples, ' +
                          'but %d samples were found!' % num_test)
        if num_train + num_val != 1281167:
            warnings.warn('ILSVRC2012 should contain 1,281,167 training ' +
                          'samples, but %d samples were found!'
                          % (num_train + num_val))

        # Maximum string length of an image path.
        max_path_len = len(max(self._torch_ds_train.samples +
            ([] if num_val == 0 else self._torch_ds_val.samples) +
            self._torch_ds_test.samples, key=lambda t : len(t[0]))[0])

        self._data['classification'] = True
        self._data['sequence'] = False
        self._data['num_classes'] = 1000
        self._data['is_one_hot'] = use_one_hot

        self._data['in_shape'] = [224, 224, 3]
        self._data['out_shape'] = [1000 if use_one_hot else 1]

        self._data['in_data'] = np.chararray([num_samples, 1],
            itemsize=max_path_len, unicode=True)
        for i, (img_path, _) in enumerate(self._torch_ds_train.samples +
                ([] if num_val == 0 else self._torch_ds_val.samples) +
                self._torch_ds_test.samples):
            self._data['in_data'][i, :] = img_path

        labels = np.array(self._torch_ds_train.targets +
                          ([] if num_val == 0 else self._torch_ds_val.targets) +
                          self._torch_ds_test.targets).reshape(-1, 1)
        if use_one_hot:
            labels = self._to_one_hot(labels)
        self._data['out_data'] = labels

        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train + num_val, num_samples)
        if num_val == 0:
            self._data['val_inds'] = None
        else:
            self._data['val_inds'] = np.arange(num_train, num_train + num_val)

        print('Dataset consists of %d training, %d validation and %d test '
              % (num_train, num_val, num_test) + 'samples.')

        end = time.time()
        print('Elapsed time to read dataset: %f sec' % (end-start))

    def tf_input_map(self, mode='inference'):
        """Not impemented."""
        # Confirm, whether you wanna process data as in the baseclass or
        # implement a new image loader.
        raise NotImplementedError('Not implemented yet!')

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'ILSVRC2012'

    def _read_meta(self, meta_fn):
        """Read the meta file to know how to translate WNID to ILSVRC2012_ID.

        The following attributes are added (dictionaries):
            _imid_to_wnid: ILSVRC2012_ID to WNID.
            _wnid_to_imid: WNID to ILSVRC2012_ID.
            _imid_to_words: ILSVRC2012_ID to set of words (textual description
                of label).

        Args:
            meta_fn: Path to meta file.
        """
        meta = loadmat(meta_fn)['synsets']

        # ILSVRC2012_ID -> WNID
        imid2wnid = dict()
        # ILSVRC2012_ID -> Example Words
        imid2words = dict()

        for i in range(meta.size):
            imid = meta[i][0][0][0][0] # ILSVRC2012_ID
            wnid = meta[i][0][1][0] # WNID
            words = meta[i][0][2][0] # words
            num_children = meta[i][0][4][0][0]
            if num_children != 0:
                # We don't care about non-leaf nodes.
                assert(imid >= 1000)
                continue
            assert(imid >= 1 and imid <= 1000)

            # NOTE internally, we subtract 1 from all ILSVRC2012_ID to have
            # labels between 0 and 999.
            imid2wnid[imid-1] = wnid
            imid2words[imid-1] = words

        assert(len(imid2wnid.keys()) == 1000)

        wnid2imid = {v: k for k, v in imid2wnid.items()}
        assert(len(wnid2imid.keys()) == 1000)

        self._imid_to_wnid = imid2wnid
        self._wnid_to_imid = wnid2imid
        self._imid_to_words = imid2words

    def _process_dataset(self, train_dir, val_dir, use_one_hot,
                         num_val_per_class):
        """Read and process the datasets using PyTorch its ImageFolder class.

        The labels used by the ImageFolder class are changed to match the
        ILSVRC2012_ID labels (where 1 is subtracted to get labels between 0 and
        999).

        Additionally, this method splits the Imagenet training set into train
        and validation set. The original ImageNet validation set is used as test
        set.

        The following attributes are added to the class:
            _torch_ds_train: A PyTorch Dataset class representing the training
                set.
            _torch_ds_test: A PyTorch Dataset class representing the validation
                set (corresponds to the dataset in "val_dir").
            _torch_ds_val: A PyTorch Dataset class representing the validation
                set (A subset of the training set).
            _wnid_to_clbl: A dictionary translating WNID to the "common label",
                that is used by data loaders that simply use the "ImageFolder"
                class. For instance, the pretrained ImageNet classifiers in
                the the PyTorch model zoo:
                    https://pytorch.org/docs/stable/torchvision/models.html

        Args:
            See docstring of constructor.
            train_dir: Path to ILSVRC2012 training images.
            val_dir: Path to ILSVRC2012 validation images.
        """
        # Read raw dataset using the PyTorch ImageFolder class.
        train_transform, test_transform = \
            ILSVRC2012Data.torch_input_transforms()
        ds_train = datasets.ImageFolder(train_dir, train_transform)
        ds_test = datasets.ImageFolder(val_dir, test_transform)
        ds_val = None

        ### Translate targets to ILSVRC2012_ID labels.
        wnid2lbl = ds_train.class_to_idx
        # Sanity check.
        assert(len(wnid2lbl.keys()) == len(ds_test.class_to_idx.keys()))
        for k in wnid2lbl.keys():
            assert(k in ds_test.class_to_idx.keys())
            assert(wnid2lbl[k] == ds_test.class_to_idx[k])

        lbl2wnid = {v: k for k, v in wnid2lbl.items()}

        for ds_obj in [ds_train, ds_test]:
            for s in range(len(ds_obj.samples)):
                img_path, lbl = ds_obj.samples[s]
                assert(ds_obj.targets[s] == lbl)

                wnid = lbl2wnid[lbl]
                # We assume a folder structure where images are stored under
                # their corresponding WNID.
                assert(wnid in img_path)

                imid = self._wnid_to_imid[wnid]

                ds_obj.samples[s] = (img_path, imid)
                ds_obj.targets[s] = imid

                assert(ds_obj.imgs[s][1] == imid)

            # The mapping from class name (WNID) to label has changed!
            ds_obj.class_to_idx = self._wnid_to_imid

        ### Split training set into train/val set.
        if num_val_per_class > 0:
            orig_samples = ds_train.samples
            ds_train.samples = None
            ds_train.imgs = None
            ds_train.targets = None

            ds_val = deepcopy(ds_train)
            ds_val.transform = test_transform
            assert(ds_val.target_transform is None)

            num_classes = len(self._imid_to_wnid.keys())
            assert(num_classes == 1000)
            val_counts = np.zeros(num_classes, dtype=np.int)

            ds_train.samples = []
            ds_train.imgs = ds_train.samples
            ds_val.samples = []
            ds_val.imgs = ds_val.samples

            for img_path, img_lbl in orig_samples:
                if val_counts[img_lbl] >= num_val_per_class: # train sample
                    ds_train.samples.append((img_path, img_lbl))
                else: # validation sample
                    val_counts[img_lbl] += 1
                    ds_val.samples.append((img_path, img_lbl))

            ds_train.targets = [s[1] for s in ds_train.samples]
            ds_val.targets = [s[1] for s in ds_val.samples]

            for ds_obj in [ds_train, ds_val]:
                assert(len(ds_obj.samples) == len(ds_obj.imgs) and \
                       len(ds_obj.samples) == len(ds_obj.targets))

        self._torch_ds_train = ds_train
        self._torch_ds_test = ds_test
        self._torch_ds_val = ds_val
        
        self._wnid_to_clbl = wnid2lbl

    def to_common_labels(self, outputs):
        """Translate between label conventions.

        Translate a given set of labels (that correspond to the
        ``ILSVRC2012_ID`` (minus one) of their images) back to the labels
        provided by the :class:`torchvision.datasets.ImageFolder` class.

        Note:
            This would be the label convention for ImageNet used by
            PyTorch examples.

        Args:
            outputs: Targets (as integers or 1-hot encodings).

        Returns:
            The translated targets (if the targets where given as 1-hot
            encodings, then this method also returns 1-hot encodings).
        """
        is_np = False
        # We don't want to do inplace modifications.
        if isinstance(outputs, np.ndarray):
            is_np = True
            outputs = np.copy(outputs)
        else:
            assert(isinstance(outputs, torch.Tensor))
            outputs = outputs.clone()

        is_1_hot = False
        if len(outputs.shape) == 2 and outputs.shape[1] == self.num_classes:
            if not is_np:
                raise NotImplementedError('Method can\'t deal with 1-hot ' +
                    'encodings provided as Torch tensors yet!')
            is_1_hot = True
            outputs = self._to_one_hot(outputs, reverse=True)

        for i in range(outputs.shape[0]):
            wnid = self._imid_to_wnid[int(outputs[i])]
            outputs[i] = self._wnid_to_clbl[wnid]

        if is_1_hot:
            outputs = self._to_one_hot(outputs, reverse=False)

        return outputs

    @staticmethod
    def torch_input_transforms():
        """Get data augmentation pipelines for ILSVRC2012 inputs.

        Note, the augmentation is inspired by the augmentation proposed in:
            https://git.io/fjWPZ

        Returns:
            (tuple): Tuple containing:

                - **train_transform**: A transforms pipeline that applies random
                  transformations, normalizes the image and resizes/crops it
                  to a final size of 224 x 224 pixels.
                - **test_transform**: Similar to train_transform, but no random
                  transformations are applied.
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        return train_transform, test_transform

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.
        
        Note, label ID in the plot correspond to ``ILSVRC2012_ID`` minus 1.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("ILSVRC2012 Sample")
        else:
            assert(np.size(outputs) == 1)
            label = np.asscalar(outputs)
            label_name = self._imid_to_words[label]

            if predictions is None:
                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label))
            else:
                if np.size(predictions) == self.num_classes:
                    pred_label = np.argmax(predictions)
                else:
                    pred_label = np.asscalar(predictions)
                pred_label_name = self._imid_to_words[pred_label]

                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label) + '\nPrediction: %s (%d)' % \
                             (pred_label_name, pred_label))

        if inputs.size == 1:
            img = self.read_images(inputs)
        else:
            img = inputs

        ax.set_axis_off()
        ax.imshow(np.squeeze(np.reshape(img, self.in_shape)))
        fig.add_subplot(ax)

if __name__ == '__main__':
    pass


