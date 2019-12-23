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
@title           :tests/test_utils.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/14/2019
@version         :1.0
@python_version  :3.6.8

Helper functions that are useful when implementing unit tests.
"""

import contextlib
import io
import sys
import inspect
import unittest


@contextlib.contextmanager
def nostdout():
    """A context that can be used to surpress all std outputs of a function.

    The code from this method has been copied from (accessed: 08/14/2019):
        https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto

    NOTE Our copyright and license does not apply for this function.
    We use this code WITHOUT ANY WARRANTIES.

    Instead, the code in this method is licensed under CC BY-SA 3.0:
        https://creativecommons.org/licenses/by-sa/3.0/

    The code stems from an answer by Alex Martelli:
        https://stackoverflow.com/users/95810/alex-martelli

    The answer has been editted by Nick T:
        https://stackoverflow.com/users/194586/nick-t
    """
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def unittest_verbosity():
    """Return the verbosity setting of the currently running unittest
    program, or 0 if none is running.
    
    The code from this method has been copied from (accessed: 08/14/2019):
        https://stackoverflow.com/questions/13761697/how-to-access-the-unittest-mainverbosity-setting-in-a-unittest-testcase

    NOTE Our copyright and license does not apply for this function.
    We use this code WITHOUT ANY WARRANTIES.

    Instead, the code in this method is licensed under CC BY-SA 3.0:
        https://creativecommons.org/licenses/by-sa/3.0/

    The code stems from an answer by Gareth Rees:
        https://stackoverflow.com/users/68063/gareth-rees
    """
    frame = inspect.currentframe()
    while frame:
        self = frame.f_locals.get('self')
        if isinstance(self, unittest.TestProgram):
            return self.verbosity
        frame = frame.f_back
    return 0

if __name__ == '__main__':
    pass


