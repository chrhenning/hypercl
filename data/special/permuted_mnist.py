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
# @title           :permuted_mnist.py
# @author          :ch
# @contact         :henningc@ethz.ch
# @created         :04/11/2019
# @version         :1.0
# @python_version  :3.6.7
"""
Permuted MNIST Dataset
^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.special.permuted_mnist` contains a data handler for the
permuted MNIST dataset.
"""
import copy

from data.mnist_data import MNISTData

class PermutedMNISTList():
    """A list of permuted MNIST tasks that only uses a single instance of class
    :class:`PermutedMNIST`.

    An instance of this class emulates a Python list that holds objects of
    class :class:`PermutedMNIST`. However, it doesn't actually hold several
    objects, but only one with just the permutation matrix being exchanged
    everytime a different element of this list is retrieved. Therefore, **use
    this class with care**!

        - As all list entries are the same PermutedMNIST object, one should
          never work with several list entries at the same time!
          -> **Retrieving a new list entry will modify every previously
          retrieved list entry!**
        - **Everytime a list entry is retrieved, the batch generator of the
          underlying PermutedMNIST object is reset**, hence training and test
          indices start from the beginning.
        - When retrieving a slice, a shallow copy of this object is created
          (i.e., the underlying :class:`PermutedMNIST` does not change) with
          only the desired subgroup of permutations avaliable.

    Why would one use this object? When working with many permuted MNIST tasks,
    then the memory consumption becomes significant if one desires to hold all
    task instances at once in working memory. An object of this class only needs
    to hold the MNIST dataset once in memory. Just the number of permutation
    matrices grows linearly with the number of tasks.

    Caution:
        **You may never use more than one entry of this class at the same
        time**, as all entries share the same underlying data object and
        therewith the same permutation.

    Example:
        You should **never** use this list as follows

        .. code-block:: python

            dhandlers = PermutedMNISTList(permutations, '/tmp')
            d0 = dhandlers[0]
            # Zero-th permutation is active ...
            # ...
            d1 = dhandlers[1]
            # First permutation is active for `d0` and `d1`!
            # Important, you may not use `d0` anymore, as this might lead to
            # undesired behavior.

    Example:
        Instead, always work with only one list entry at a time. The following
        usage would be **correct**

        .. code-block:: python

            dhandlers = PermutedMNISTList(permutations, '/tmp')
            d = dhandlers[0]
            # Zero-th permutation is active ...
            # ...
            d = dhandlers[1]
            # First permutation is active for `d` as expected.

    Args:
        (....): See docstring of constructor of class :class:`PermutedMNIST`.
        permutations: A list of permutations (see parameter ``permutation``
            of class :class:`PermutedMNIST` to have a description of valid list
            entries). The length of this list denotes the number of tasks.
        show_perm_change_msg: Whether to print a notification everytime the
            data permutation has been exchanged. This should be enabled
            during developement such that a proper use of this list is
            ensured. **Note** You may never work with two elements of this
            list at a time.
    """
    def __init__(self, permutations, data_path, use_one_hot=True,
                 validation_size=0, padding=0, show_perm_change_msg=True):
        print('Loading MNIST into memory, that is shared among %d permutation '
              % (len(permutations)) + 'tasks.')

        self._data = PermutedMNIST(data_path, use_one_hot=use_one_hot,
            validation_size=validation_size, permutation=None, padding=padding)

        self._permutations = permutations

        self._show_perm_change_msg = show_perm_change_msg

    def __len__(self):
        """Number of tasks."""
        return len(self._permutations)

    def __getitem__(self, index):
        """Return the underlying data object with the index'th permutation.

        Args:
            index: Index of task for which data should be returned.

        Return:
            The data loader for task ``index``.
        """
        ### User Warning ###
        color_start = '\033[93m'
        color_end = '\033[0m'
        help_msg = 'To disable this message, disable the flag ' + \
            '"show_perm_change_msg" when calling the constructor of class ' + \
            'classifier.permuted_mnist.PermutedMNISTList.'
        ####################

        if isinstance(index, slice):
            new_list = copy.copy(self)
            new_list._permutations = self._permutations[index]

            ### User Warning ###
            if self._show_perm_change_msg:
                indices = list(range(*index.indices(len(self))))
                print(color_start + 'classifier.permuted_mnist.' +
                      'PermutedMNISTList: A slice of permutations with ' +
                      'indices %s has been created. ' % indices +
                      'The applied permutation has not changed! ' + color_end +
                      help_msg)
            ####################

            return new_list

        assert(isinstance(index, int))
        self._data.permutation = self._permutations[index]
        self._data.reset_batch_generator()

        ### User Warning ###
        if self._show_perm_change_msg:
            color_start = '\033[93m'
            color_end = '\033[0m'

            print(color_start + 'classifier.permuted_mnist.PermutedMNISTList:' +
                  ' Data permutation has been changed to %d. ' % index +
                  color_end + help_msg)
        ####################

        return self._data

    def __setitem__(self, key, value):
        """Not implemented."""
        raise NotImplementedError('Not yet implemented!')

    def __delitem__(self, key):
        """Not implemented."""
        raise NotImplementedError('Not yet implemented!')

class PermutedMNIST(MNISTData):
    """An instance of this class shall represent the permuted MNIST dataset,
    which is the same as the MNIST dataset, just that input pixels are shuffled
    by a random matrix.

    Note:
        Image transformations are computed on the fly when transforming batches
        to torch tensors. Hence, this class is only applicable to PyTorch
        applications. Internally, the class stores the unpermuted images.

    Attributes:
        permutation: The permuation matrix that is applied to input images
            before they are transformed to Torch tensors.
        torch_in_shape: The input shape of images, similar to attribute
            `in_shape`. In contrast to `in_shape`, this attribute reflects the
            padding that is applied when calling
            :meth:`classifier.permuted_mnist.PermutedMNIST.\
input_to_torch_tensor`.

    Args:
        data_path: Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot: Whether the class labels should be represented in a
            one-hot encoding.
        validation_size: The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
        permutation: The permutation that should be applied to the dataset.
            If ``None``, no permutation will be applied. We expect a numpy
            permutation of the form
            :code:`np.random.permutation((28+2*padding)**2)`
        padding: The amount of padding that should be applied to images.

            .. note::
                The padding is currently not reflected in the
                `:attr:`data.dataset.Dataset.in_shape` attribute, as the padding
                is only applied to torch tensors. See attribute
                :attr:`torch_in_shape`.
    """
    def __init__(self, data_path, use_one_hot=True, validation_size=0,
                 permutation=None, padding=0):
        super().__init__(data_path, use_one_hot=use_one_hot,
                         validation_size=validation_size)

        self._padding = padding
        self._input_dim = (28+padding*2)**2

        self.permutation = permutation # See setter below.

    @property
    def permutation(self):
        """Getter for attribute :attr:`permutation`"""
        return self._permutation

    @permutation.setter
    def permutation(self, value):
        """Setter for the attribute :attr:`permutation`."""
        self._permutation = value
        self._transform = PermutedMNIST.torch_input_transforms(
            padding=self._padding, permutation=value)

    @property
    def torch_in_shape(self):
        """Getter for attribute :attr:`torch_in_shape`"""
        return [self.in_shape[0] + 2 * self._padding,
                self.in_shape[1] + 2 * self._padding, self.in_shape[2]]

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'PermutedMNIST'

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        Note, this method has been overwritten from the base class.

        It applies zero padding and pixel permutations.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor.
        """
        if not force_no_preprocessing:
            assert (len(x.shape) == 2)  # batch size plus flattened image.

            from torch import stack

            img_size = 28 + 2*self._padding

            # Transform the numpy data into a representation as expected by the
            # ToPILImage transformation.
            x = (x * 255.0).astype('uint8')
            x = x.reshape(-1, 28, 28, 1)

            x = stack([self._transform(x[i, ...]) for i in
                       range(x.shape[0])]).to(device)

            # Transform tensor back to numpy shape.
            # FIXME This is a horrible solution, but at least we ensure that the
            # user gets a tensor in the same shape as always and does not have to
            # deal with cases.
            x = x.permute(0, 2, 3, 1)
            x = x.contiguous().view(-1, img_size**2)

            return x

        else:
            return MNISTData.input_to_torch_tensor(self, x, device, mode=mode,
                force_no_preprocessing=force_no_preprocessing)

    @staticmethod
    def torch_input_transforms(permutation=None, padding=0):
        """Transform MNIST images to PyTorch tensors.

        Args:
            permutation: A given permutation that should be applied to all
                images.
            padding: Apply a given amount of zero padding.

        Returns:
            A transforms pipeline.
        """
        import torchvision.transforms as transforms

        # The following functions has been copied and modified from:
        #   https://git.io/fjqzP
        # Note, that a different license and copyright applies and that we use
        # this code WITHOUT ANY WARRANTIES.
        """
        MIT License

        Copyright (c) 2018 Gido van de Ven

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """
        def _permutate_image_pixels(image, permutation):
            '''Permutate the pixels of an image according to 'permutation'.

            Args:
                image: 3D-tensor containing the image
                permutation: <ndarray> of pixel-indices in their new order

            Returns:
                Permuted image.
            '''

            if permutation is None:
                return image
            else:
                c, h, w = image.size()
                image = image.view(c, -1)
                image = image[:,
                        permutation]  # --> same permutation for each channel
                image = image.view(c, h, w)

            return image

        transform = transforms.Compose([
            transforms.ToPILImage('L'),
            transforms.Pad(padding),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: _permutate_image_pixels(x, permutation)),
        ])

        return transform

    def tf_input_map(self, mode='inference'):
        """Not implemented! The class currently does not support Tensorflow."""
        # FIXME Permutations are applied on the fly when images are translated
        # PyTorch tensors. Internally, we store normal MNIST images.
        raise NotImplementedError('No Tensorflow support for this class ' +
                                  'implemented.')

if __name__ == '__main__':
    pass
