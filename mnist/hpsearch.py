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
- **title**          :hpsearch/hpsearch.py
- **author**         :ch
- **contact**        :henningc@ethz.ch
- **created**        :05/05/2019
- **version**        :1.0
- **python_version** :3.6.8

A very simple hyperparameter search. The results will be gathered as a CSV file.

Here is an example on how to start an hyperparameter search on a cluster using
:code:`bsub`:

.. code-block:: console

   $ bsub -n 1 -W 48:00 -e hpsearch.err -o hpsearch.out \\
     -R "rusage[mem=8000]" \\
     python3 hpsearch.py --run_cluster --num_jobs=20

For more demanding jobs (e.g., ImageNet), one may request more resources:

.. code-block:: console

   $ bsub -n 1 -W 96:00 -e hpsearch.err -o hpsearch.out \\
     -R "rusage[mem=16000]" \\
     python3 hpsearch.py --run_cluster --num_jobs=20 --num_hours=48 \\
     --resources="\"rusage[mem=8000, ngpus_excl_p=1]\""

Please fill in the grid parameters in the corresponding config file (see
command line argument `grid_module`).
"""
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import

import argparse
import random
import warnings
import numpy as np
import os
from datetime import datetime
from subprocess import call
import pandas
import re
import time
import pickle
import traceback
import sys
import importlib

from utils import misc

# From which module to read the default grid.
_DEFAULT_GRID = 'mnist.hp_search_splitMNIST'

### The following variables will be otherwritten in the main ###
################################################################
### See method `_read_config`.
# Name of the script that should be executed by the hyperparameter search.
# Note, the working directory is set seperately by the hyperparameter search
# script.
_SCRIPT_NAME = None # Has to be specified in helper module!
# This file is expected to reside in the output folder of the simulation.
_SUMMARY_FILENAME = None # Has to be specified in helper module!
# These are the keywords that are supposed to be in the summary file.
# A summary file always has to include the keyword "finished"!.
_SUMMARY_KEYWORDS = None # Has to be specified in helper module!
# The name of the command-line argument that determines the output folder
# of the simulation.
_OUT_ARG = 'out_dir' # Default value if attribute `_OUT_ARG` does not exist.
# Function handle to parser of performance summary file.
_SUMMARY_PARSER_HANDLE = None # Default parser `_get_performance_summary` used.
# A function handle, that is used to evaluate whether an output folder should
# be kept.
_PERFORMANCE_EVAL_HANDLE = None # Has to be set in config file.
# According to which keyword will the CSV be sorted.
_PERFORMANCE_KEY = None # First key in `_SUMMARY_KEYWORDS` will be used.
# Sort order.
_PERFORMANCE_SORT_ASC = False
# FIXME should be deleted soon.
_ARGPARSE_HANDLE = None
################################################################

# This will be a list of booleans, each representing whether a specific cmd has
# been executed.
_CMD_FINISHED = None

def _grid_to_commands(grid_dict):
    """Translate a dictionary of parameter values into a list of commands.

    Args:
        grid_dict: A dictionary of argument names to lists, where each list
            contains possible values for this argument.

    Returns:
        A list of dictionaries. Each key is an argument name that maps onto a
        single value.
    """
    # We build a list of dictionaries with key value pairs.
    commands = []

    # We need track of the index within each value array.
    gkeys = list(grid_dict.keys())
    indices = [0] * len(gkeys)

    stopping_criteria = False
    while not stopping_criteria:

        cmd = dict()
        for i, k in enumerate(gkeys):
            v = grid_dict[k][indices[i]]
            cmd[k] = v
        commands.append(cmd)
        
        for i in range(len(indices)-1,-1,-1):
            indices[i] = (indices[i] + 1) % len(grid_dict[gkeys[i]])
            if indices[i] == 0 and i == 0:
                stopping_criteria = True
            elif indices[i] != 0:
                break

    return commands

def _args_to_cmd_str(cmd_dict, out_dir=None):
    """Translate a dictionary of argument names to values into a string that
    can be typed into a console.

    Args:
        cmd_dict: Dictionary with argument names as keys, that map to a value.
        out_dir (optional): The output directory that should be passed to the
            command. No output directory will be passed if not specified.

    Returns:
        A string of the form:
            python3 train.py --out_dir=OUT_DIR --ARG1=VAL1 ...
    """
    cmd_str = 'python3 %s' % _SCRIPT_NAME

    if out_dir is not None:
        cmd_str += ' --%s=%s' % (_OUT_ARG, out_dir)

    for k, v in cmd_dict.items():
        if type(v) == bool:
            cmd_str += ' --%s' % k if v else ''
        else:
            cmd_str += ' --%s=%s' % (k, str(v))

    return cmd_str

def _get_performance_summary(out_dir, cmd_ident):
    """Parse the performance summary file of a simulation.

    This is a very primitive parser, that expects that each line of the
    result file :code:`os.path.join(out_dir, _SUMMARY_FILENAME)` is a
    keyword-value pair. The keyword is taken from the :code:`_SUMMARY_KEYWORDS`
    list. **They must appear in the correct order.**
    The value can either be a single number or a list of numbers. A list of
    numbers will be converted into a string, such that it appears in a single
    cell under the given keyword when opening the result CSV file with a
    spreadsheet.

    Args:
        out_dir: The output directory of the simulation.
        cmd_ident (int): Identifier of this command (needed for informative
            error messages).

    Raises:
        IOError: If performance summary file does not exist.
        ValueError: If a summary key is not at the expected position in the
            result file.

    Returns:
        A dictionary containing strings as keywords. Note, the values may not be
        lists, and strings need to be wrapped into an extra layer of double
        quotes such that the spreadsheet interprets them as a single entity.
    """
    # Get training results.
    result_summary_fn = os.path.join(out_dir, _SUMMARY_FILENAME)
    if not os.path.exists(result_summary_fn):
        raise IOError('Training run %d did not finish. No results!' \
                      % (cmd_ident+1))

    with open(result_summary_fn, 'r') as f:
        result_summary = f.readlines()

    # Ensure downwards compatibility!
    summary_keys = _SUMMARY_KEYWORDS

    performance_dict = dict()
    for line, key in zip(result_summary, summary_keys):
        if not line.startswith(key):
            raise ValueError('Key %s does not appear in performance '
                             % (key) + 'summary where it is expected.')
        # Parse the lines of the result file.
        # Crop keyword to retrieve only the value.
        _, line = line.split(' ', maxsplit=1)
        # https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
        line_nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(line_nums) == 1: # Single number
            performance_dict[key] = [line_nums[0]]
        else: # List of numbers
            # Convert list to a string for the resulting CSV file. Note, the
            # quotes are needed that the list will be written into a single cell
            # when opening the csv file (note, every key can have exactly one
            # value).
            performance_dict[key] = \
                ['"' + misc.list_to_str(line_nums, delim=',') + '"']

    return performance_dict

def _run_cmds_on_single_machine(args, commands, out_dir, results_file):
    """Method to run the jobs sequentially on a single machine.

    Args:
        args: Command-line arguments.
        commands: List of command dictionaries.
        out_dir: Output directory.
        results_file: CSV file to store summary.
    """
    # FIXME: The code in this function is mostly a dublicate of the code in
    # function _run_cmds_on_cluster.

    for i, cmd_dict in enumerate(commands):
        # A call might fail for several reasons. We don't want the whole search
        # to fail.
        try:
            # FIXME quick and dirty solution.
            cmd_out_dir = os.path.join(out_dir,
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
            if os.path.exists(cmd_out_dir):
                time.sleep(1.1)
                cmd_out_dir = os.path.join(out_dir,
                    datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
            assert(not os.path.exists(cmd_out_dir))

            cmd_str = _args_to_cmd_str(cmd_dict, out_dir=cmd_out_dir)
            cmd_dict[_OUT_ARG] = cmd_out_dir

            # Execute the program.
            print('Starting training run %d/%d -- "%s"' % (i+1, len(commands),
                                                            cmd_str))
            ret = call(cmd_str, shell=True)
            print('Call finished with return code %d.' % ret)

            try:
                # We store the command used for execution. This might be helpful
                # for the user in case he wants to manually continue the
                # simulation.
                with open(os.path.join(cmd_out_dir, 'hpsearch_command.sh'),
                          'w') as f:
                    f.write('#!/bin/sh\n')
                    f.write('%s' % (_args_to_cmd_str(cmd_dict)))

                # Get training results.
                performance_dict = _SUMMARY_PARSER_HANDLE(cmd_out_dir, i)
                for k, v in performance_dict.items():
                    cmd_dict[k] = v

                # Create or update the CSV file summarizing all runs.
                panda_frame = pandas.DataFrame.from_dict(cmd_dict)
                if os.path.isfile(results_file):
                    old_frame = pandas.read_csv(results_file, sep=';')
                    panda_frame = pandas.concat([old_frame, panda_frame],
                                                sort=True)
                panda_frame.to_csv(results_file, sep=';', index=False)

                # Check whether simulation has finished successfully.
                has_finished = int(cmd_dict['finished'][0])
                if has_finished == 1:
                    _CMD_FINISHED[i] = True
                else:
                    _CMD_FINISHED[i] = False

            except Exception:
                traceback.print_exc(file=sys.stdout)
                warnings.warn('Could not assess whether run %d has been ' \
                                % (i+1) + 'completed.')

        except Exception:
            traceback.print_exc(file=sys.stdout)
            warnings.warn('Call %d/%d failed -- "%s".' % (i+1, len(commands), \
                _args_to_cmd_str(cmd_dict)))

def _run_cmds_on_cluster(args, commands, out_dir, results_file):
    """This method will submit a certain number of jobs onto an LSF cluster and
    wait for these jobs to complete before starting new jobs. This allows to
    run several jobs in parallel.

    Args:
        args: Command-line arguments.
        commands: List of command dictionaries.
        out_dir: Output directory.
        results_file: CSV file to store summary.
    """
    from bsub import bsub

    def check_running(jobs):
        rjobs = bsub.running_jobs()
        tmp_jobs = jobs
        jobs = []
        for job, cmd_dict, ind in tmp_jobs:
            if job.job_id in rjobs:
                jobs.append((job, cmd_dict, ind))
                continue

            print('Job %d finished.' % ind)
            cmd_out_dir = cmd_dict[_OUT_ARG]

            try:
                # We store the command used for execution. This might be helpful
                # for the user in case he wants to manually continue the
                # simulation.
                with open(os.path.join(cmd_out_dir, 'hpsearch_command.sh'),
                          'w') as f:
                    f.write('#!/bin/sh\n')
                    f.write('%s' % (_args_to_cmd_str(cmd_dict)))

                # Get training results.
                performance_dict = _SUMMARY_PARSER_HANDLE(cmd_out_dir, i)
                for k, v in performance_dict.items():
                    cmd_dict[k] = v

                # Create or update the CSV file summarizing all runs.
                panda_frame = pandas.DataFrame.from_dict(cmd_dict)
                if os.path.isfile(results_file):
                    old_frame = pandas.read_csv(results_file, sep=';')
                    panda_frame = pandas.concat([old_frame, panda_frame],
                                                sort=True)
                panda_frame.to_csv(results_file, sep=';', index=False)

                # Check whether simulation has finished successfully.
                has_finished = int(cmd_dict['finished'][0])
                if has_finished == 1:
                    _CMD_FINISHED[ind] = True
                else:
                    _CMD_FINISHED[ind] = False

            except Exception:
                traceback.print_exc(file=sys.stdout)
                warnings.warn('Could not assess whether run %d has been ' \
                              % (ind+1) + 'completed.')

        return jobs

    jobs = []
    i = -1
    while len(commands) > 0:
        jobs = check_running(jobs)
        while len(jobs) >= args.num_jobs:
            time.sleep(10)
            jobs = check_running(jobs)

        cmd_dict = commands.pop()
        i += 1

        # FIXME quick and dirty solution.
        folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
        if os.path.exists(os.path.join(out_dir, folder_name)):
            time.sleep(1.1)
            folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
        cmd_out_dir = os.path.join(out_dir, folder_name)
        assert(not os.path.exists(cmd_out_dir))

        cmd_str = _args_to_cmd_str(cmd_dict, out_dir=cmd_out_dir)
        cmd_dict[_OUT_ARG] = cmd_out_dir

        # Execute the program.
        print('Starting training run %d/%d -- "%s"' % (i+1, len(commands),
                                                       cmd_str))

        job_name = 'job_%s' % folder_name
        # FIXME the bsub module ignores the pathnames we set. Hence, all output
        # files are simply stored in the local directory. For now, we will
        # capture this in the postprocessing script.
        job_error_file = os.path.join(cmd_out_dir, job_name + '.err')
        job_out_file = os.path.join(cmd_out_dir, job_name + '.out')
        sub = bsub(job_name, R=args.resources, n=1, W='%d:00' % args.num_hours,
                   e=job_error_file, o=job_out_file, verbose=True)
        sub(cmd_str)
        jobs.append((sub, cmd_dict, i))

    # Wait for all jobs to complete.
    while len(jobs) > 0:
        time.sleep(10)
        jobs = check_running(jobs)

def _backup_commands(commands, out_dir):
    """This function will generate a bash script that resembles the order in
    which the individual commands have been executed. This is important, as the
    order might be random. This script is just another helper for the user to
    follow the execution order. Additionally, this file save the commands as
    pickle. This is a backup for future usage (i.e., maybe a continue search
    option will be build in at some point).

    Args:
        commands: List of command dictionaries.
        out_dir: Output directory.
    """
    fn_script = os.path.join(out_dir, 'commands.sh')
    fn_pickle = os.path.join(out_dir, 'commands.pickle')

    with open(fn_pickle, 'wb') as f:
        pickle.dump(commands, f)

    with open(fn_script, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('# This script contains all %d commands that are planned ' \
                % (len(commands)) + 'to be executed during this ' +
                'hyperparameter search. The order of execution is preserved ' +
                'in this script.\n')
        f.write('# Note, output directories are not yet specified in the ' +
                'commands listed here!\n')
        for cmd in commands:
            f.write('%s\n\n' % (_args_to_cmd_str(cmd)))

def _store_incomplete(commands, out_dir):
    """This function will pickle all command dictionaries of commands that have
    not been completed. This might be used to just continue an interrupted
    hyperparameter search.

    Args:
        commands: List of command dictionaries.
        out_dir: Output directory.
    """
    incomplete = []

    for i, cmd in enumerate(commands):
        if not _CMD_FINISHED[i]:
            incomplete.append(cmd)

    if len(incomplete) == 0:
        return

    warnings.warn('%d runs have not been completed.' % (len(incomplete)))
    fn_pickle = os.path.join(out_dir, 'not_completed.pickle')
    with open(fn_pickle, 'wb') as f:
        pickle.dump(incomplete, f)

def _read_config(config_mod, require_perf_eval_handle=False,
                 require_argparse_handle=False):
    """Parse the configuration module and check whether all attributes are set
    correctly.

    This function will set the corresponding global variables from this script
    appropriately.

    Args:
        config_mod: The implemented configuration template
            :mod:`hpsearch.hpsearch_postprocessing`.
        require_perf_eval_handle: Whether :attr:`_PERFORMANCE_EVAL_HANDLE` has
            to be specified in the config file.
        require_argparse_handle: Whether :attr:`_ARGPARSE_HANDLE` has to be
            specified in the config file.
    """
    assert(hasattr(config_mod, '_SCRIPT_NAME'))
    assert(hasattr(config_mod, '_SUMMARY_FILENAME'))
    assert(hasattr(config_mod, '_SUMMARY_KEYWORDS') and \
           'finished' in config_mod._SUMMARY_KEYWORDS)
    globals()['_SCRIPT_NAME'] = config_mod._SCRIPT_NAME
    globals()['_SUMMARY_FILENAME'] = config_mod._SUMMARY_FILENAME
    globals()['_SUMMARY_KEYWORDS'] = config_mod._SUMMARY_KEYWORDS

    # Ensure downwards compatibility -- attributes did not exist previously.
    if hasattr(config_mod, '_OUT_ARG'):
        globals()['_OUT_ARG'] = config_mod._OUT_ARG

    if hasattr(config_mod, '_SUMMARY_PARSER_HANDLE') and \
            config_mod._SUMMARY_PARSER_HANDLE is not None:
        globals()['_SUMMARY_PARSER_HANDLE'] = config_mod._SUMMARY_PARSER_HANDLE
    else:
        globals()['_SUMMARY_PARSER_HANDLE'] = _get_performance_summary

    if require_perf_eval_handle:
        assert(hasattr(config_mod, '_PERFORMANCE_EVAL_HANDLE') and \
               config_mod._PERFORMANCE_EVAL_HANDLE is not None)
        globals()['_PERFORMANCE_EVAL_HANDLE'] = \
            config_mod._PERFORMANCE_EVAL_HANDLE
    else:
        if not hasattr(config_mod, '_PERFORMANCE_EVAL_HANDLE') or \
                config_mod._PERFORMANCE_EVAL_HANDLE is None:
            warnings.warn('Attribute "_PERFORMANCE_EVAL_HANDLE" not defined ' +
                          'in configuration file but might be required in ' +
                          'future releases.')

    if hasattr(config_mod, '_PERFORMANCE_KEY') and \
            config_mod._PERFORMANCE_KEY is not None:
        globals()['_PERFORMANCE_KEY'] = config_mod._PERFORMANCE_KEY
    else:
        globals()['_PERFORMANCE_KEY'] = config_mod._SUMMARY_KEYWORDS[0]

    if hasattr(config_mod, '_PERFORMANCE_SORT_ASC'):
        globals()['_PERFORMANCE_SORT_ASC'] = config_mod._PERFORMANCE_SORT_ASC

    if require_argparse_handle:
        assert(hasattr(config_mod, '_ARGPARSE_HANDLE') and \
               config_mod._ARGPARSE_HANDLE is not None)
        globals()['_ARGPARSE_HANDLE'] = config_mod._ARGPARSE_HANDLE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= \
        'hpsearch - Automatic Parameter Search -- ' +
        'Note, that the search values are defined in the source code of the ' +
        'accompanied configuration file!')
    parser.add_argument('--deterministic_search', action='store_true',
                        help='If not selected, the order of configurations ' +
                             'is randomly picked.')
    parser.add_argument('--num_searches', type=int, metavar='N', default=-1,
                        help='If not -1, then the number of configurations ' +
                             'that should be tested maximally. ' +
                             'Default: %(default)s.')
    parser.add_argument('--out_dir', type=str,
                        default='./out/hyperparam_search',
                        help='Where should all the output files be written ' +
                             'to? Default: %(default)s.')
    parser.add_argument('--grid_module', type=str, default=_DEFAULT_GRID,
                        help='Name of module to import from which to read ' +
                             'the hyperparameter search grid. The module ' +
                             'must define the two variables "grid" and ' +
                             '"conditions". Default: %(default)s.')
    parser.add_argument('--run_cwd', type=str, default='.',
                        help='The working directory in which runs are ' +
                             'executed (in case the run script resides at a ' +
                             'different folder than this hpsearch script. ' +
                             'All outputs of this script will be relative to ' +
                             'this working directory (if output folder is ' +
                             'defined as relative folder). ' +
                             'Default: "%(default)s".')
    parser.add_argument('--run_cluster', action='store_true',
                        help='This option would produce jobs for a GPU ' +
                             'cluser running the IBM LSF batch system.')
    parser.add_argument('--num_jobs', type=int, metavar='N', default=8,
                        help='If "run_cluster" is activated, then this ' +
                             'option determines the maximum number of jobs ' +
                             'that can be submitted in parallel. ' +
                             'Default: %(default)s.')
    parser.add_argument('--num_hours', type=int, metavar='N', default=24,
                        help='If "run_cluster" is activated, then this ' +
                             'option determines the maximum number of hours ' +
                             'a job may run on the cluster. ' +
                             'Default: %(default)s.')
    parser.add_argument('--resources', type=str,
                        default='"rusage[mem=8000, ngpus_excl_p=1]"',
                        help='If "run_cluster" is activated, then this ' +
                             'option determines the resources assigned to ' +
                             'job in the hyperparameter search (option -R ' +
                             'of bsub). Default: %(default)s.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=42,
                        help='Random seed. Default: %(default)s.')
    # TODO build in "continue" option to finish incomplete commands.
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    ### Get hyperparameter search grid from specified module.
    grid_module = importlib.import_module(args.grid_module)
    assert(hasattr(grid_module, 'grid') and hasattr(grid_module, 'conditions'))
    grid = grid_module.grid
    conditions = grid_module.conditions

    assert(len(grid) > 0)

    _read_config(grid_module)

    print('### Running Hyperparameter Search ...')

    if len(conditions) > 0:
        print('Note, %d conditions have been defined and will be enforced!' % \
              len(conditions))

    if args.run_cwd != '.':
        os.chdir(args.run_cwd)
        print('Current working directory: %s.' % os.path.abspath(os.curdir))

    ### Output directory creation.
    out_dir = os.path.join(args.out_dir,
        'search_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print('Results will be stored in %s.' % os.path.abspath(out_dir))
    # FIXME we should build in possibilities to merge with previous searches.
    assert(not os.path.exists(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ### Build the grid.
    # We build a list of dictionaries with key value pairs.
    commands = _grid_to_commands(grid)

    # Ensure, that all conditions can be enforced.
    for i, cond in enumerate(conditions):
        for k in cond[0].keys():
            if k not in grid.keys():
                warnings.warn('Condition %d can not be enforced. ' % (i) +
                    'Key %s is not specified in grid -- %s.' % (k, str(cond)))

    # Now, we have the commands according to the grid, but we still need to
    # enforce the conditions.
    # This list will keep track of the conditions each command is affected.
    for i, cond_tup in enumerate(conditions):
        cond, action = cond_tup
        cond_keys = list(cond.keys())

        affected = [False] * len(commands)
        new_commands = []
        for j, command in enumerate(commands):
            # Figure out, whether condition i is satisfied for command j.
            comm_keys = command.keys()
            key_satisfied = [False] * len(cond_keys)
            for l, cond_key in enumerate(cond_keys):
                if cond_key in comm_keys:
                    cond_vals = cond[cond_key]
                    if command[cond_key] in cond_vals:
                        key_satisfied[l] = True

            if np.all(key_satisfied):
                affected[j] = True
            else:
                continue

            # Generate a set of replacement commands for command j, such that
            # condition i is satisfied.
            cmds = _grid_to_commands(action)
            for l, cmd in enumerate(cmds):
                for k in comm_keys:
                    if k not in cmd.keys():
                        cmds[l][k] = command[k]
            new_commands.extend(cmds)

        # Remove all commands affected by this condition and insert the new
        # ones.
        old_cmds = commands
        commands = []
        for j, cmd in enumerate(old_cmds):
            if not affected[j]:
                commands.append(cmd)
        commands.extend(new_commands)

    # Note, the way we enforced conditions above may result in dublicates.
    # We need to remove them now.
    old_cmds = commands
    commands = []
    for i in range(len(old_cmds)):
        cmd_i = old_cmds[i]
        has_dublicate = False
        for j in range(i+1, len(old_cmds)):
            cmd_j = old_cmds[j]

            if len(cmd_i.keys()) != len(cmd_j.keys()):
                continue

            is_dublicate = True
            for k in cmd_i.keys():
                if k not in cmd_j.keys():
                    is_dublicate = False
                    break
                if cmd_i[k] != cmd_j[k]:
                    is_dublicate = False
                    break

            if is_dublicate:
                has_dublicate = True
                break

        if not has_dublicate:
            commands.append(cmd_i)

    ### Random shuffling of command execution order.
    if not args.deterministic_search:
        random.shuffle(commands)

    ### Consider the maximum number of commands we may execute.
    if args.num_searches != -1 and len(commands) > args.num_searches:
        print('Only %d of %d configurations will be tested!' % \
              (args.num_searches, len(commands)))
        commands = commands[:args.num_searches]

    ### Print all commands to user to allow visual verification.
    print('\n### List of all commands. Please verify carefully. ###\n')
    for cmd in commands:
        print(_args_to_cmd_str(cmd))
    print('\nThe %d command(s) above will be executed.' % len(commands))
    _CMD_FINISHED = [False] * len(commands)

    # The list of command strings will be dumped into a file, such that the
    # user sees their order.
    _backup_commands(commands, out_dir)

    ### Hyperparameter Search
    # Where do we summarize the results?
    results_file = os.path.join(out_dir, 'search_results.csv')

    try:
        if not args.run_cluster:
            _run_cmds_on_single_machine(args, commands, out_dir, results_file)
        else:
            _run_cmds_on_cluster(args, commands, out_dir, results_file)
    except Exception:
        traceback.print_exc(file=sys.stdout)
        warnings.warn('An error occurred during the hyperparameter search.')

    _store_incomplete(commands, out_dir)

    ### Sort CSV file according to performance key.
    csv_file_content = pandas.read_csv(results_file, sep=';')
    csv_file_content = csv_file_content.sort_values(_PERFORMANCE_KEY,
        ascending=_PERFORMANCE_SORT_ASC)
    csv_file_content.to_csv(results_file, sep=';', index=False)

    print('### Running Hyperparameter Search ... Done')
