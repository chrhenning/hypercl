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
"""
@title           :classifier/permuted_mnist.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :04/11/2019
@version         :1.0
@python_version  :3.6.7

A data handler for a permuted MNIST dataset.
"""
from data.mnist_data import MNISTData

class PermutedMNIST(MNISTData):
    """An instance of this class shall represent the permuted MNIST dataset,
    which is the same as the MNIST dataset, just that input pixels are shuffled
    by a random matrix.

    Attributes: (additional to baseclass)
    """

    def __init__(self, data_path, use_one_hot=True, validation_size=0,
                 permutation=None, padding=0):
        """Read the MNIST digit classification dataset from file.

        Image transformations are computed on the fly when transforming batches
        to torch tensors. Hence, this class is only applicable to PyTorch
        applications. Internally, the class stores the unpermuted images.

        Args:
            data_path: Where should the dataset be read from? If not existing,
                the dataset will be downloaded into this folder.
            use_one_hot (default: False): Whether the class labels should be
                represented in a one-hot encoding.
            validation_size: The number of validation samples. Validation
                samples will be taking from the training set (the first n
                samples).
            permutation: The permutation that should be applied to the dataset.
                If None, no permutation will be applied. We expect a numpy
                permutation of the form
                    np.random.permutation((28+2*padding)**2)
            padding: The amount of padding that should be applied to images.
        """
        super().__init__(data_path, use_one_hot=use_one_hot,
                         validation_size=validation_size)

        self._padding = padding
        self._input_dim = (28+padding*2)**2
        self._permutation = permutation
        self._transform = PermutedMNIST.torch_input_transforms(padding=padding,
            permutation=permutation)

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'PermutedMNIST'

    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        It applies zero padding and pixel permutations.

        Args:
            x: A 2D numpy array, containing inputs as provided by this dataset.
            device: The PyTorch device onto which the input should be mapped.
            mode: This is the same as the mode attribute in the class
                  NetworkBase, that can be used to distinguish between training
                  and inference (e.g., if special input processing should be
                  used during training).
                  Valid values are: 'train' and 'inference'.
            force_no_preprocessing: In case preprocessing is applied to the
                inputs (e.g., normalization or random flips/crops), this option
                can be used to prohibit any kind of manipulation. Hence, the
                inputs are transformed into PyTorch tensors on an "as is" basis.

        Returns:
            The given input x as PyTorch tensor.
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

if __name__ == '__main__':
    pass
