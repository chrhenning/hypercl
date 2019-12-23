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
@title           :test_train.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/13/2019
@version         :1.0
@python_version  :3.6.8

The major goal of these test cases is to ensure that the performance of the
toy regression does not change while this repo is under developement.
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import unittest
import sys
import tempfile
import os
import shutil
import contextlib
import time

from toy_example import train
#from tests. test_utils import nostdout
from tests.test_utils import unittest_verbosity

class TrainTestCase(unittest.TestCase):
    def setUp(self):
        pass # Nothing to setup.

    def test_cl_hnet_setup(self):
        """This method tests whether the CL capabilities of the 3 polynomials
        toy regression remain as reported in the readme of the corresponding
        folder."""
        verbosity_level = unittest_verbosity()
        targets = [0.004187723621726036, 0.002387890825048089,
                   0.006071540527045727]

        # Without timestamp, test would get stuck/fail if someone mistakenly
        # starts the test case twice.
        timestamp = int(time.time() * 1000)
        out_dir = os.path.join(tempfile.gettempdir(),
                               'test_cl_hnet_setup_%d' % timestamp)
        my_argv = ['foo', '--no_plots', '--no_cuda', '--beta=0.005',
                   '--emb_size=2', '--n_iter=4001', '--lr_hyper=1e-2',
                   '--data_random_seed=42', '--out_dir=%s' % out_dir]
        sys.argv = list(my_argv)

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        if verbosity_level == 2:
            fmse, _, _ = train.run()
        else:
            #with nostdout():
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    fmse, _, _ = train.run()
        shutil.rmtree(out_dir)

        self.assertEqual(len(fmse), len(targets))
        for i in range(len(fmse)):
            self.assertAlmostEqual(fmse[i], targets[i], places=3)

    def tearDown(self):
        pass # Nothing to clean up.

if __name__ == '__main__':
    unittest.main()


