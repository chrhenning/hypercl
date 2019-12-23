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
# @title           :data/cub_200_2011_data.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :05/17/2019
# @version         :1.0
# @python_version  :3.6.8
"""
CUB-200-2011 Dataset
--------------------

The module :mod:`data.cub_200_2011_data` contains a dataloader for the
Caltech-UCSD Birds-200-2011 Dataset (CUB-200-2011).

The dataset is available at:

    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

For more information on the dataset, please refer to the corresponding
publication:

    Wah et al., "The Caltech-UCSD Birds-200-2011 Dataset",
    California Institute of Technology, 2011.
    http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf

The dataset consists of 11,788 images divided into 200 categories. The dataset
has a specified train/test split and a lot of additional information (bounding
boxes, segmentation, parts annotation, ...) that we don't make use of yet.

.. note::
    This dataset should not be confused with the older version CUB-200,
    containing only 6,033 images.

.. note ::
    We use the same data augmentation as for class
    :class:`data.ilsvrc2012_data.ILSVRC2012Data`.

.. note::
    The original category labels range from 1-200. We modify them to range
    from 0 - 199.
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

import torchvision.datasets as datasets
import os
import time
import urllib.request
import tarfile
import pandas
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from data.large_img_dataset import LargeImgDataset
from data.ilsvrc2012_data import ILSVRC2012Data

class CUB2002011(LargeImgDataset):
    """An instance of the class shall represent the CUB-200-2011 dataset.

    The input data of the dataset will be strings to image files. The output
    data corresponds to object labels (bird categories).

    Note:
        The dataset will be downloaded if not available.

    Note:
        The original category labels range from 1-200. We modify them to
        range from 0 - 199.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be represented in a
            one-hot encoding.

            .. note::
                This option does not influence the internal PyTorch
                Dataset classes (e.g., cmp.
                :attr:`data.large_img_dataset.LargeImgDataset.torch_train`),
                that can be used in conjunction with PyTorch data loaders.
        num_val_per_class (int): The number of validation samples per class.
            For instance: If value 10 is given, a validation set of size
            5 * 200 = 1,000 is constructed (these samples will be removed
            from the training set).

            .. note::
                Validation samples use the same data augmentation pipeline
                as test samples.
    """
    _DOWNLOAD_PATH = 'http://www.vision.caltech.edu/visipedia-data/' + \
                     'CUB-200-2011/'
    _IMG_ANNO_FILE = 'CUB_200_2011.tgz'
    _SEGMENTATION_FILE = 'segmentations.tgz' # UNUSED
    # In which subfolder of the datapath should the data be stored.
    _SUBFOLDER = 'cub_200_2011'
    # After extracting the downloaded archive, the data will be in
    # this subfolder.
    _REL_BASE = 'CUB_200_2011'
    _IMG_DIR = 'images' # Realitve to _REL_BASE
    _CLASSES_FILE = 'classes.txt' # Realitve to _REL_BASE
    _IMG_CLASS_LBLS_FILE = 'image_class_labels.txt' # Realitve to _REL_BASE
    _IMG_FILE = 'images.txt' # Realitve to _REL_BASE
    _TRAIN_TEST_SPLIT_FILE = 'train_test_split.txt' # Realitve to _REL_BASE

    def __init__(self, data_path, use_one_hot=False, num_val_per_class=0):
        # We keep the full path to each image in memory, so we don't need to
        # tell the super class the root path to each image (i.e., samples
        # contain absolute not relative paths).
        super().__init__('')

        start = time.time()

        print('Reading CUB-200-2011 dataset ...')

        # Actual data path
        data_path = os.path.join(data_path, CUB2002011._SUBFOLDER)

        if not os.path.exists(data_path):
            print('Creating directory "%s" ...' % (data_path))
            os.makedirs(data_path)
            
        full_data_path = os.path.join(data_path, CUB2002011._REL_BASE)
        image_dir = os.path.join(full_data_path, CUB2002011._IMG_DIR)
        classes_fn = os.path.join(full_data_path, CUB2002011._CLASSES_FILE)
        img_class_fn = os.path.join(full_data_path,
                                    CUB2002011._IMG_CLASS_LBLS_FILE)
        image_fn = os.path.join(full_data_path, CUB2002011._IMG_FILE)
        train_test_split_fn = os.path.join(full_data_path,
                                           CUB2002011._TRAIN_TEST_SPLIT_FILE)

        ########################
        ### Download dataset ###
        ########################
        if not os.path.exists(image_dir) or \
                not os.path.exists(classes_fn) or \
                not os.path.exists(img_class_fn) or \
                not os.path.exists(image_fn) or \
                not os.path.exists(train_test_split_fn):
            print('Downloading dataset ...')
            archive_fn = os.path.join(data_path, CUB2002011._IMG_ANNO_FILE)
            urllib.request.urlretrieve(CUB2002011._DOWNLOAD_PATH + \
                                       CUB2002011._IMG_ANNO_FILE, \
                                       archive_fn)
            # Extract downloaded dataset.
            tar = tarfile.open(archive_fn, "r:gz")
            tar.extractall(path=data_path)
            tar.close()

            os.remove(archive_fn)

        ####################
        ### Read dataset ###
        ####################
        # We use the same transforms as 
        train_transform, test_transform = \
            ILSVRC2012Data.torch_input_transforms()
        # Consider all images as training images. We split the dataset later.
        ds_train = datasets.ImageFolder(image_dir, train_transform)

        # Ability to translate image IDs into image paths and back.
        image_ids_csv = pandas.read_csv(image_fn, sep=' ',
                                        names=['img_id', 'img_path'])
        id2img = dict(zip(list(image_ids_csv['img_id']),
                          list(image_ids_csv['img_path'])))
        # Since the ImageFolder class uses absolute paths, we have to change
        # the just read relative paths.
        for iid in id2img.keys():
            id2img[iid] = os.path.join(image_dir, id2img[iid])
        img2id = {v: k for k, v in id2img.items()}

        # Image ID to label.
        img_lbl_csv = pandas.read_csv(img_class_fn, sep=' ',
                                      names=['img_id', 'label'])
        id2lbl = dict(zip(list(img_lbl_csv['img_id']),
                          list(img_lbl_csv['label'])))
        # Note, categories go from 1-200. We change them to go from 0 - 199.
        for iid in id2lbl.keys():
            id2lbl[iid] = id2lbl[iid] - 1

        # Image ID to label name.
        img_lbl_name_csv = pandas.read_csv(classes_fn, sep=' ',
                                           names=['label', 'label_name'])
        lbl2lbl_name_tmp = dict(zip(list(img_lbl_name_csv['label']),
                                    list(img_lbl_name_csv['label_name'])))
        # Here, we also have to modify the labels to be within 0-199.
        lbl2lbl_name = {k-1: v for k, v in lbl2lbl_name_tmp.items()}

        # Train-test-split.
        train_test_csv = pandas.read_csv(train_test_split_fn, sep=' ',
                                         names=['img_id', 'is_train'])
        id2train = dict(zip(list(train_test_csv['img_id']),
                            list(train_test_csv['is_train'])))

        self._label_to_name = lbl2lbl_name

        ####################
        ### Sanity check ###
        ####################
        for i, (img_path, lbl) in enumerate(ds_train.samples):
            iid = img2id[img_path]
            assert(id2img[iid] == img_path)
            assert(lbl == id2lbl[iid])

        ################################
        ### Train / val / test split ###
        ################################
        orig_samples = ds_train.samples
        ds_train.samples = []
        ds_train.imgs = ds_train.samples
        ds_train.targets = []

        ds_test = deepcopy(ds_train)
        ds_test.transform = test_transform
        assert(ds_test.target_transform is None)
        if num_val_per_class > 0:
            ds_val = deepcopy(ds_train)
            # NOTE we use test input transforms for the validation set.
            ds_val.transform = test_transform
        else:
            ds_val = None

        num_classes = len(lbl2lbl_name_tmp.keys())
        assert(num_classes == 200)
        val_counts = np.zeros(num_classes, dtype=np.int)

        for img_path, img_lbl in orig_samples:
            iid = img2id[img_path]
            if id2train[iid] == 1: # In train split.
                if val_counts[img_lbl] >= num_val_per_class: # train sample
                    ds_train.samples.append((img_path, img_lbl))
                else: # validation sample
                    val_counts[img_lbl] += 1
                    ds_val.samples.append((img_path, img_lbl))
            else: # In test split.
                ds_test.samples.append((img_path, img_lbl))

        for ds_obj in [ds_train, ds_test] + \
                ([ds_val] if num_val_per_class > 0 else []):
            ds_obj.targets = [s[1] for s in ds_obj.samples]
            assert(len(ds_obj.samples) == len(ds_obj.imgs) and \
                   len(ds_obj.samples) == len(ds_obj.targets))

        self._torch_ds_train = ds_train
        self._torch_ds_test = ds_test
        self._torch_ds_val = ds_val

        #####################################
        ### Build internal data structure ###
        #####################################
        num_train = len(self._torch_ds_train)
        num_test = len(self._torch_ds_test)
        num_val = 0 if self._torch_ds_val is None else \
            len(self._torch_ds_val)
        num_samples = num_train + num_test + num_val

        max_path_len = len(max(orig_samples, key=lambda t : len(t[0]))[0])

        self._data['classification'] = True
        self._data['sequence'] = False
        self._data['num_classes'] = 200
        self._data['is_one_hot'] = use_one_hot

        self._data['in_shape'] = [224, 224, 3]
        self._data['out_shape'] = [200 if use_one_hot else 1]

        self._data['in_data'] = np.chararray([num_samples, 1],
            itemsize=max_path_len, unicode=True)
        for i, (img_path, _) in enumerate(ds_train.samples +
                ([] if num_val == 0 else ds_val.samples) +
                ds_test.samples):
            self._data['in_data'][i, :] = img_path

        labels = np.array(ds_train.targets +
                          ([] if num_val == 0 else ds_val.targets) +
                          ds_test.targets).reshape(-1, 1)
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

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'CUB-200-2011'

    def tf_input_map(self, mode='inference'):
        """Not impemented."""
        # Confirm, whether you wanna process data as in the baseclass or
        # implement a new image loader.
        raise NotImplementedError('Not implemented yet!')

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("CUB-200-2011 Sample")
        else:
            assert(np.size(outputs) == 1)
            label = np.asscalar(outputs)
            label_name = self._label_to_name[label]

            if predictions is None:
                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label))
            else:
                if np.size(predictions) == self.num_classes:
                    pred_label = np.argmax(predictions)
                else:
                    pred_label = np.asscalar(predictions)
                pred_label_name = self._label_to_name[pred_label]

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


