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
# @title          :cifar/train_zenke.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :12/12/2019
# @version        :1.0
# @python_version :3.6.8
"""
CIFAR-10/100 Zenkenet Experiment
---------------------------------

The script  :mod:`cifar.train_zenke` is used to run a CL experiment on CIFAR
using the Zenkenet (:class:`mnets.zenkenet.ZenkeNet`). At the moment, it simply
takes care of providing the correct command-line arguments and default values to
the end user. Afterwards, it will simply call: :mod:`cifar.train`.

See :ref:`cifar-readme-zenke-reference-label` for usage instructions.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

from cifar import train_args
from cifar import train

if __name__ == '__main__':
    config = train_args.parse_cmd_arguments(mode='zenke_cifar')

    train.run(config, experiment='zenke')


