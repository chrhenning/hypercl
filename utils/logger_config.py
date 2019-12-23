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
"""
@title           :logger_config.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/09/2018
@version         :1.0
@python_version  :3.6.6

Collection of methods used to setup and maintain the logger used by this
framework.
"""

import logging
import os
import sys

def config_logger(name, log_file, file_level, console_level):
    """Configure the logger that should be used by all modules in this
    package.
    This method sets up a logger, such that all messages are written to console
    and to an extra logging file. Both outputs will be the same, except that
    a message logged to file contains the module name, where the message comes
    from.

    The implementation is based on an earlier implementation of a function I
    used in another project:

        https://git.io/fNDZJ

    Args:
        name: The name of the created logger.
        log_file: Path of the log file. If None, no logfile will be generated.
            If the logfile already exists, it will be overwritten.
        file_level: Log level for logging to log file.
        console_level: Log level for logging to console.

    Returns:
        The configured logger.
    """
    file_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s' \
                                       + ' - %(module)s - %(message)s', \
                                       datefmt='%m/%d/%Y %I:%M:%S %p')
    stream_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s' \
                                         + ' - %(message)s', \
                                         datefmt='%m/%d/%Y %I:%M:%S %p')

    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir != '' and not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        if os.path.exists(log_file):
            os.remove(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(file_level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_level)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if log_file is not None:
        logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

if __name__ == '__main__':
    pass
