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
# @title           :data/large_img_dataset.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :09/20/2018
# @version         :1.0
# @python_version  :3.6.6
"""
Wrapper for large image datasets
--------------------------------

The module :mod:`data.large_img_dataset` contains an abstract wrapper for large
datasets, that have images as inputs. Typically, these datasets are too large to
be loaded into memory. Though, their outputs (labels) can still easily be hold
in memory. Hence, the idea is, that instead of loading the actual images, we
load the paths for each image into memory. Then we can load the images from disk
as needed.

To sum up, handlers that implement this interface will hold the outputs and
paths for the input images of the whole dataset in memory, but not the actual
images.

As an alternative, one can implement wrappers for HDF5 and TFRecord files.

Here is a simple example that illustrates the format of the dataset:

    https://www.tensorflow.org/guide/datasets#decoding_image_data_and_resizing_\
it

In case of working with PyTorch, rather than using the internal methods for
batch processing (such as :meth:`data.dataset.Dataset.next_train_batch`) one
should adapt PyTorch its data processing utilities (consisting of
:class:`torch.utils.data.Dataset` and :class:`torch.utils.data.DataLoader`)
in combination with class attributes such as
:attr:`data.large_img_dataset.LargeImgDataset.torch_train`.
"""
import numpy as np
import os
#import matplotlib.image as mpimg
from PIL import Image

from data.dataset import Dataset

class LargeImgDataset(Dataset):
    """A general dataset template for datasets with images as inputs, that are
    locally stored as individual files. Note, that this is an abstract class
    that should not be instantiated.

    Hints, when implementing the interface:

        - Attribute :attr:`data.dataset.Dataset.in_shape` still has to be
          correctly implemented, independent of the fact, that the actual input
          data is a list of strings.

    Attributes:
        imgs_path (str): The base path of all images.
        png_format_used (bool): Whether png or jped encoded of images is
            assumed.
        torch_train (torch.utils.data.Dataset): The PyTorch compatible training
            dataset.
        torch_test (torch.utils.data.Dataset): The PyTorch compatible test
            dataset.
        torch_val (torch.utils.data.Dataset): The PyTorch compatible validation
            dataset.

    Args:
        imgs_path (str): The path to the folder, containing the image files
            (the actual image paths contained in the input data (see e.g.,
            :meth:`data.dataset.Dataset.get_train_inputs`) will
            be concatenated to this path).
        png_format (bool): The images are typically assumed to be jpeg encoded.
            You may change this to png enocded images.
    """
    def __init__(self, imgs_path, png_format=False):
        super().__init__()

        self._imgs_path = imgs_path
        self._png_format_used = png_format

        # The wrapper is currently not meant for sequence inputs. You can still
        # set this variable to true, if you have sequence outputs.
        self._data['sequence'] = False

        # Implementing classes should provide instances of class
        #   torch.utils.data.Dataset
        # For instance, using torchvision.datasets.ImageFolder.
        # In this way, users can reuse typical PyTorch code and don't have to
        # write custom data loaders.
        self._torch_ds_train = None
        self._torch_ds_test = None
        self._torch_ds_val = None

    @property
    def imgs_path(self):
        """Getter for read-only attribute :attr:`imgs_path`."""
        return self._imgs_path

    @property
    def png_format_used(self):
        """Getter for read-only attribute :attr:`png_format_used`."""
        return self._png_format_used

    @property
    def torch_train(self):
        """Getter for read-only attribute :attr:`torch_train`."""
        if self._torch_ds_train is None:
            raise NotImplementedError('Dataset not prepared for PyTorch use!')
        return self._torch_ds_train

    @property
    def torch_test(self):
        """Getter for read-only attribute :attr:`torch_test`."""
        if self._torch_ds_test is None:
            raise NotImplementedError('Dataset not prepared for PyTorch use!')
        return self._torch_ds_test

    @property
    def torch_val(self):
        """Getter for read-only attribute :attr:`torch_val`."""
        return self._torch_ds_val

    def get_train_inputs(self):
        """Get the inputs of all training samples.
        
        Returns:
            (numpy.chararray): An np.chararray, where each row corresponds to an
            image file name.
        """
        return Dataset.get_train_inputs(self)

    def get_test_inputs(self):
        """Get the inputs of all test samples.
        
        Returns:
            (numpy.chararray): An np.chararray, where each row corresponds to an
            image file name.
        """
        return Dataset.get_test_inputs(self)

    def get_val_inputs(self):
        """Get the inputs of all validation samples.
        
        Returns:
            (numpy.chararray): An np.chararray, where each row corresponds to an
            image file name. If no validation set exists, ``None`` will be
            returned.
        """
        return Dataset.get_val_inputs(self)

    def read_images(self, inputs):
        """For the given filenames, read and return the images.

        Args:
            inputs (numpy.chararray): An np.chararray of filenames.

        Returns:
            (numpy.ndarray): A 2D numpy array, where each row contains a
            picture.
        """
        ret = np.empty([inputs.shape[0], np.prod(self.in_shape)], np.float32)

        for i in range(inputs.shape[0]):
            fn = os.path.join(self.imgs_path, 
                              str(inputs[i, np.newaxis].squeeze()))
            img = Image.open(fn)
            #img = mpimg.imread(fn)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(self.in_shape[:2], Image.BILINEAR)
            ret[i, :] = np.array(img).flatten()

        # Note, the images already have pixel values between 0 and 1 for
        # PNG images.
        if not self.png_format_used:
            ret /= 255.

        return ret

    def tf_input_map(self, mode='inference'):
        """Note, this method has been overwritten from the base class.

        It provides a function handle that loads images from file, resizes them
        to match the internal input image size and then flattens the image to
        a vector.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.tf_input_map`.

        Returns:
            (function): A function handle, that maps the given input tensor to
            the preprocessed input tensor.
        """
        # FIXME removed this statement from the top of this file because it
        # caused warnings when using the logging module.
        import tensorflow as tf

        base_path = os.path.join(self.imgs_path, '')
        
        def load_inputs(inputs):
            filename = tf.add(base_path, tf.squeeze(inputs))
            image_string = tf.read_file(filename)
            if self.png_format_used:
                image = tf.image.decode_png(image_string)
            else:
                image = tf.image.decode_jpeg(image_string)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_images(image, self.in_shape[:2])
            # We always feed flattened images into the network.
            image = tf.reshape(image, [-1])

            return image

        return load_inputs

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False):
        """Note, this method has been overwritten from the base class. It should
        not be used for large image datasets. Instead, the class should provide
        instances of class :class:`torch.utils.data.Dataset` for training,
        validation and test set:

            - :attr:`torch_train`
            - :attr:`torch_test`
            - :attr:`torch_val`
        """
        raise NotImplementedError('Use attributes "torch_train", "torch_val" ' +
            'and "torch_test" instead. Please refer to the class ' +
            'documentation.')

if __name__ == '__main__':
    pass


