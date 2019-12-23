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
# @title           :data/celeba_data.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :09/20/2018
# @version         :1.0
# @python_version  :3.6.6
"""
CelebA Dataset
--------------

The module :mod:`data.celeba_data` contains a handler for the CelebA dataset.

More information about the dataset can be retrieved from:
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Note, in the current implementation, this handler will not download and extract
the dataset for you. You have to do this manually by following the instructions
of the README file (which is located in the same folder as this file).

**Note, this dataset has not yet been prepared for PyTorch use!**
"""

import numpy as np
import os
import sys
import time
from collections import defaultdict
import matplotlib.pyplot as plt

from data.large_img_dataset import LargeImgDataset

class CelebAData(LargeImgDataset):
    """An instance of the class shall represent the CelebA dataset.

    The input data of the dataset will be strings to image files. The output
    data will be vectors of booleans, denoting whether a certain type of
    attribute is present in the picture.

    .. note::
        The dataset has to be already downloaded and extracted before
        this class can be instantiated. See the local README file for details.

    Args:
        data_path (str): Where should the dataset be read from?
        use_png (bool): Whether the png rather than the jpeg images should be
            used. Note, this class only considers the aligned and cropped
            images.
        shape (optional): If given, this images loaded from disk will be
            reshaped to that shape.
    """
    # Folder containing dataset.
    _ROOT = 'CelebA'
    _ANNO = os.path.join(_ROOT, 'Anno')
    _EVAL = os.path.join(_ROOT, 'Eval')
    _IMG = os.path.join(_ROOT, 'Img')
    _IDENTITY = os.path.join(_ANNO, 'identity_CelebA.txt')
    _ATTRIBUTES = os.path.join(_ANNO, 'list_attr_celeba.txt')
    # What are the bounding boxes of the faces in the original images.
    # Note, they don't match the aligned and cropped images. These images where
    # extracted by an (to me) unknown algorithm that rotated the images along
    # the eyeline and then cropped them to a size of 218*178.
    # So, these bounding boxes are basically useless.
    _BBOX = os.path.join(_ANNO, 'list_bbox_celeba.txt')
    # The landmarks in the original images.
    _ORIG_LANDMARKS = os.path.join(_ANNO, 'list_landmarks_celeba.txt')
    _LANDMARKS = os.path.join(_ANNO, 'list_landmarks_align_celeba.txt')
    _PARTITIONS = os.path.join(_EVAL, 'list_eval_partition.txt')
    _ORIG_IMGS = os.path.join(_IMG, 'img_celeba.7z')
    _PNG_IMGS = os.path.join(_IMG, 'img_align_celeba_png.7z')
    _JPG_IMGS = os.path.join(_IMG, 'img_align_celeba')
    
    def __init__(self, data_path, use_png=False, shape=None):
        if use_png:
            imgs_path = os.path.join(data_path, CelebAData._PNG_IMGS)
        else:
            imgs_path = os.path.join(data_path, CelebAData._JPG_IMGS)
        super().__init__(imgs_path, use_png)

        start = time.time()

        print('Reading CelebA dataset ...')

        # Actual data path
        root_path = os.path.join(data_path, CelebAData._ROOT)
        anno_path = os.path.join(data_path, CelebAData._ANNO)
        eval_path = os.path.join(data_path, CelebAData._EVAL)

        # TODO Download and extract the data.

        err_msg = 'Please follow the steps described in the file ' + \
            'data/README.md to download and extract the data.'
        if not os.path.exists(root_path):
            raise FileNotFoundError('Dataset not found in directory ' +
                root_path + '.\n' + err_msg)
        elif not os.path.exists(anno_path):
            raise FileNotFoundError('Annotations not found in directory ' +
                anno_path + '.\n' + err_msg)
        elif not os.path.exists(eval_path):
            raise FileNotFoundError('Partitioning not found in directory ' +
                eval_path + '.\n' + err_msg)
        elif not os.path.exists(imgs_path):
            raise FileNotFoundError('Images not found in directory ' +
                imgs_path + '.\n' + err_msg)

        try:
            self._read_dataset(data_path, shape=shape)
        except:
            print(sys.exc_info()[0])
            raise Exception('Could not read dataset. Maybe the dataset is not '
                            + 'correctly prepared.\n' + err_msg)

        end = time.time()
        print('Elapsed time to read dataset: %f sec' % (end-start))

    def _read_dataset(self, data_path, shape=None):
        """Read the dataset into memory. Note, the images are not fetched into
        memory, only their filenames.

        Args:
            data_path: Where is the relative location of the dataset.
            shape (optional): The shape of the input images.
        """
        def to_img_path(filename):
            """The image filenames from file have to be converted, if the png
            format is used.
            """
            if self.png_format_used:
                filename = os.path.splitext(filename)[0] + '.png'
            return filename
        
        # FIXME If we use the attributes as outputs, then this is a multi-label
        # classification task. Though, we don't capture this case in the
        # Dataset base class yet (it would destroy the current implementation
        # of one hot encodings).
        self._data['classification'] = False
        self._data['sequence'] = False
        self._data['num_classes'] = 40 # 40 different attributes.
        self._data['is_one_hot'] = False
        if shape is not None:
            assert(len(shape) == 2)
            self._data['in_shape'] = shape + [3]
        else:
            self._data['in_shape'] = [218, 178, 3]
        self._data['out_shape'] = [self._data['num_classes']]
        
        self._data['celeba'] = dict()

        # The annotations dict will contain the annotations of each image
        # except its attributes (i.e., the stuff we currently don't use).
        annotations = defaultdict(dict)

        ## Identity
        # Read the identities. Images with the same identity show the same
        # person.
        ident_fn = os.path.join(data_path, CelebAData._IDENTITY)
        with open(ident_fn) as f:
            ident_file = f.readlines()

        for line in ident_file:
            img_ident, ident = line.split()
            img_ident = to_img_path(img_ident)
            annotations[img_ident]['ident'] = int(ident)

        # Initialize the actual data arrays.
        num_imgs = len(annotations.keys())
        max_str_len = len(max(annotations.keys(), key=len))
        in_data = np.chararray([num_imgs, 1], itemsize=max_str_len,
                               unicode=True)
        out_data = np.empty([num_imgs, self._data['num_classes']],
                            dtype=np.float32)

        ## Attributes
        # Read the list of attributes. This will become the output of this
        # dataset.
        attr_fn =  os.path.join(data_path, CelebAData._ATTRIBUTES)

        with open(attr_fn) as f:
            nis = int(f.readline())
            attr_names = f.readline().split()
            attribute_lines = f.readlines()

        assert(nis == num_imgs)
        assert(len(attr_names) == self._data['num_classes'])
        self._data['celeba']['attr_names'] = attr_names

        assert(len(attribute_lines) == num_imgs)
        for i, line in enumerate(attribute_lines):
            words = line.split()
            img_ident = to_img_path(words[0])
            attrs = [int(i) > 0 for i in words[1:]]
            assert(len(attrs) == self._data['num_classes'])

            # The actual index of the sample in the dataset.
            annotations[img_ident]['index'] = i

            ### Fill input and output data.
            in_data[i, :] = img_ident
            out_data[i, :] = attrs

        self._data['in_data'] = in_data
        self._data['out_data'] = out_data

        ## Landmarks
        # Landmarks of aligned and cropped images.
        # The following landmarks are specified for each image:
        # ['lefteye', 'righteye', 'nose', 'leftmouth', 'rightmouth']
        lm_fn =  os.path.join(data_path, CelebAData._LANDMARKS)

        with open(lm_fn) as f:
            nis = int(f.readline())
            lm_names_raw = f.readline().split()
            lm_lines = f.readlines()

        assert(nis == num_imgs)
        # A landmark always consists of an x and y coordinate.
        assert(len(lm_names_raw) % 2 == 0)
        assert(len(lm_lines) == num_imgs)

        lm_names = []
        for i in range(0, len(lm_names_raw), 2):
            assert(lm_names_raw[i].endswith('_x') and \
                   lm_names_raw[i+1].endswith('_y'))
            lm_names.append(lm_names_raw[i][:-2])
        self._data['celeba']['landmark_names'] = lm_names

        for line in lm_lines:
            words = line.split()
            img_ident = to_img_path(words[0])
            locs = [int(i) for i in words[1:]]
            assert(len(locs) == len(lm_names_raw))

            lms = dict()
            for i in range(0, len(locs), 2):
                lms[lm_names[i//2]] = (locs[i], locs[i+1])
            
            annotations[img_ident]['landmarks'] = lms

        ## Partitioning
        # Load partitioning (what samples belong to train (0), test (2) and
        # val (1) set?).
        part_fn =  os.path.join(data_path, CelebAData._PARTITIONS)
        with open(part_fn) as f:
            partitions = f.readlines()

        assert(len(partitions) == num_imgs)

        train_inds = []
        test_inds = []
        val_inds = []
        for i, line in enumerate(partitions):
            img_ident, partition = line.split()
            img_ident = to_img_path(img_ident)
            partition = int(partition)

            assert(i == annotations[img_ident]['index'])
            
            if partition == 0:
                train_inds.append(i)
            elif partition == 1:
                val_inds.append(i)
            else:
                test_inds.append(i)

        self._data['train_inds'] = np.asarray(train_inds)
        self._data['test_inds'] = np.asarray(test_inds)
        self._data['val_inds'] = np.asarray(val_inds)
        assert(len(train_inds) + len(test_inds) + len(val_inds) == num_imgs)

        self._data['celeba']['anno'] = annotations

    def get_attribute_names(self):
        """Get the names of the different attributes classified by this
        dataset.

        Returns:
            (list): A list of attributes. The order of the list will have the
            same order as the output labels.
        """
        return self._data['celeba']['attr_names']

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'CelebA'

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        ax.set_title("CelebA Sample")
        # FIXME We ignore outputs and predictions for now.

        if inputs.size == 1:
            img = self.read_images(inputs)
        else:
            img = inputs

        ax.set_axis_off()
        ax.imshow(np.squeeze(np.reshape(img, self.in_shape)))
        fig.add_subplot(ax)

if __name__ == '__main__':
    pass


