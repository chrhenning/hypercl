#!/usr/bin/env python3
# Copyright 2019 Johannes von Oswald

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
# @title           :train.py
# @author          :jvo
# @contact         :voswaldj@ethz.ch
# @created         :07/08/2019
# @version         :1.0
# @python_version  :3.6.8

"""
Continual learning of permutedMNIST with hypernetworks.
-------------------------------------------------------

This script is used to run PermutedMNIST continual learning experiments.
It's role is analogous to the one of the script :mod:`mnist.train_splitMNIST`.
Start training by executing the following command:

.. code-block:: console

    $ python train_permutedMNIST.py

"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from mnist.train_splitMNIST import run

if __name__ == '__main__':
    run(mode='perm')