#!/usr/bin/env python3
# Copyright 2019 Johannes von Oswald
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
@title           :splitCIFAR100.py
@author          :jvo
@contact         :oswald@ini.ethz.ch
@created         :05/13/2019
@version         :1.0
@python_version  :3.7.3

A wrapper for data handlers of the CIFAR10/SplitCIFAR100 task.
"""
import numpy as np

from data.cifar10_data import CIFAR10Data
from data.cifar100_data import CIFAR100Data

def get_split_CIFAR_handlers(data_path, use_one_hot=True, validation_size=0,
                             use_data_augmentation=False):
    """This method will combine 1 object of the class CIFAR10Data and
    5 objects of the class SplitCIFAR100Data. 

    The SplitCIFAR100 tasks consists of 6 tasks, corresponding to the images 
    in CIFAR10 and 5 tasks correspond to the images with 
    labels [0-10], [10-20], [20-30], [30-40], [40-50].

    Args:
        data_path: Where should the CIFAR10Data,CIFAR100Data datasets
        be read from? If not existing, the dataset will be downloaded 
        into this folder.
        use_one_hot (default: True): Whether the class labels should be
            represented in a one-hot encoding.
        validation_size: The size of the validation set of each individual
            data handler.
        use_data_augmentation (optional): Note, this option currently only
            applies to input batches that are transformed using the class
            member "input_to_torch_tensor" (hence, only available for
            PyTorch).
    Returns:
        A list of data handlers, each corresponding to a SplitCIFAR100 
        object.
    """
    print('Creating data handlers for SplitCIFAR tasks ...')

    handlers = []
    handlers.append(CIFAR10Data(data_path, use_one_hot=use_one_hot,
            validation_size=validation_size,
            use_data_augmentation=use_data_augmentation))
    for i in range(0, 50, 10):
        handlers.append(SplitCIFAR100Data(data_path, 
            use_one_hot=use_one_hot, validation_size=validation_size,
            use_data_augmentation=use_data_augmentation, labels=range(i, i+10)))

    print('Creating data handlers for SplitCIFAR tasks ... Done')

    return handlers

class SplitCIFAR100Data(CIFAR100Data):
    """An instance of the class shall represent a SplitCIFAR task.

    Attributes: (additional to baseclass)
    """
    def __init__(self, data_path, use_one_hot=False, validation_size=1000,
                 use_data_augmentation=False, labels=[0, 10],
                 full_out_dim=False):
        """Read the CIFAR100 image classification dataset from file.

        This method checks whether the dataset has been read before (a pickle
        dump has been generated). If so, it reads the dump. Otherwise, it
        reads the data from scratch and creates a dump for future usage.

        Args:
            data_path: Where should the dataset be read from? If not existing,
                the dataset will be downloaded into this folder.
            use_one_hot (default: False): Whether the class labels should be
                represented in a one-hot encoding.
            validation_size: The number of validation samples. Validation
                samples will be taking from the training set (the first n
                samples).
            use_data_augmentation (optional): Note, this option currently only
                applies to input batches that are transformed using the class
                member "input_to_torch_tensor" (hence, only available for
                PyTorch).
                Note, we are using the same data augmentation pipeline as for
                CIFAR-10.
            labels: The labels that should be part of this task.
            full_out_dim: Choose the original CIFAR instead of the the new 
                task output dimension
        """
        super().__init__(data_path, use_one_hot=use_one_hot, validation_size=0,
                         use_data_augmentation=use_data_augmentation)

        K = len(labels)

        self._labels = labels

        train_ins = self.get_train_inputs()
        test_ins = self.get_test_inputs()

        train_outs = self.get_train_outputs()
        test_outs = self.get_test_outputs()

        # Get labels.
        if self.is_one_hot:
            train_labels = self._to_one_hot(train_outs, reverse=True)
            test_labels = self._to_one_hot(test_outs, reverse=True)
        else:
            train_labels = train_outs
            test_labels = test_outs

        train_labels = train_labels.squeeze()
        test_labels = test_labels.squeeze()

        train_mask = train_labels == labels[0]
        test_mask = test_labels == labels[0]
        for k in range(1, K):
            train_mask = np.logical_or(train_mask, train_labels == labels[k])
            test_mask = np.logical_or(test_mask, test_labels == labels[k])

        train_ins = train_ins[train_mask, :]
        test_ins = test_ins[test_mask, :]

        train_outs = train_outs[train_mask, :]
        test_outs = test_outs[test_mask, :]

        if validation_size > 0:
            assert (validation_size < train_outs.shape[0])
            val_inds = np.arange(validation_size)
            train_inds = np.arange(validation_size, train_outs.shape[0])

        else:
            train_inds = np.arange(train_outs.shape[0])

        test_inds = np.arange(train_outs.shape[0],
                              train_outs.shape[0] + test_outs.shape[0])

        outputs = np.concatenate([train_outs, test_outs], axis=0)
        
        if not full_out_dim:
            outputs = self.transform_outputs(outputs)
        images = np.concatenate([train_ins, test_ins], axis=0)

        ### Overwrite internal data structure. Only keep desired labels.

        # Note, we continue to pretend to be a 100 class problem, such that
        # the user has easy access to the correct labels and has the original
        # 1-hot encodings.
        if not full_out_dim:
            self._data['num_classes'] = 10
        else:
            self._data['num_classes'] = 100
        self._data['in_data'] = images
        self._data['out_data'] = outputs
        self._data['train_inds'] = train_inds
        self._data['test_inds'] = test_inds
        if validation_size > 0:
            self._data['val_inds'] = val_inds

        n_val = 0
        if validation_size > 0:
            n_val = val_inds.size

        print('Created SplitCIFAR task with labels %s and %d train, %d test '
              % (str(labels), train_inds.size, test_inds.size) +
              'and %d val samples.' % (n_val))

    def transform_outputs(self, outputs):
        """Transform the outputs from the 100D CIFAR100 dataset 
        into proper 10D labels.

        Args:
            outputs: 2D numpy array of outputs.

        Returns:
            2D numpy array of transformed outputs.
        """
        labels = self._labels
        if self.is_one_hot:
            assert(outputs.shape[1] == self._data['num_classes'])
            mask = np.zeros(self._data['num_classes'], dtype=np.bool)
            mask[labels] = True

            return outputs[:, mask]
        else:
            assert (outputs.shape[1] == 1)
            ret = outputs.copy()
            for i, l in enumerate(labels):
                ret[ret == l] = i
            return ret

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'SplitCIFAR100'

if __name__ == '__main__':
    pass


