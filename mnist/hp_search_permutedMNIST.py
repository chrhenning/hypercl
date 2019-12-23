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
- **title**          :hpsearch/hpsearch_config_template.py
- **author**         :ch
- **contact**        :henningc@ethz.ch
- **created**        :09/12/2019
- **version**        :1.0
- **python_version** :3.6.8

**Note, this is just a template for a hyperparameter configuration and not an
actual source file.**

A configuration file for our custom hyperparameter search script
:mod:`hpsearch.hpsearch`. To setup a configuration file for your simulation,
simply create a copy of this template and follow the instructions in this file
to fill all defined attributes.

Once the configuration is setup for your simulation, you simply need to modify
the fields :attr:`grid` and :attr:`conditions` to prepare a new grid search.

**Note**, if you are implementing this template for the first time, you also
have to modify the code below the "DO NOT CHANGE THE CODE BELOW" section. Normal
users may not change the code below this heading.
"""

##########################################
### Please define all parameters below ###
##########################################

grid = {
    'experiment': ["permutedMNIST"],
    'cl_scenario': [1,2,3],
    'num_tasks': [10,100],
    'infer_task_id': [True, False],
    'infer_with_entropy': [True, False],
    'dont_set_default':[False],
    # FIXME this is due to latex error on leonhard cluster
    'random_seed': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
}

"""Parameter grid for grid search.

Define a dictionary with parameter names as keys and a list of values for
each parameter. For flag arguments, simply use the values :code:`[False, True]`.
Note, the output directory is set by the hyperparameter search script.
Therefore, it always assumes that the argument `--out_dir` exists and you
**should not** add `out_dir` to this `grid`!

Example:
    .. code-block:: python

        grid = {'option1': [10], 'option2': [0.1, 0.5],
                'option3': [False, True]}

    This dictionary would correspond to the following 4 configurations:

    .. code-block:: console

        python3 SCRIPT_NAME.py --option1=10 --option2=0.1
        python3 SCRIPT_NAME.py --option1=10 --option2=0.5
        python3 SCRIPT_NAME.py --option1=10 --option2=0.1 --option3
        python3 SCRIPT_NAME.py --option1=10 --option2=0.5 --option3

If fields are commented out (missing), the default value is used.
Note, that you can specify special :attr:`conditions` below.
"""

conditions = [
    ({'infer_with_entropy': [True], 'cl_scenario': [2,3]}, 
                                                {'infer_task_id': [True]}),
    # when the encoder does not output a softmax, then we have to enforce
    # that the latent space of the autoencoder is "related" to the task_embs
]

"""Define exceptions for the grid search.

Sometimes, not the whole grid should be searched. For instance, if an `SGD`
optimizer has been chosen, then it doesn't make sense to search over multiple
`beta2` values of an Adam optimizer.
Therefore, one can specify special conditions or exceptions.
Note* all conditions that are specified here will be enforced. Thus, **they
overwrite the** :attr:`grid` **options above**.

How to specify a condition? A condition is a key value tuple: whereas as the
key as well as the value is a dictionary in the same format as in the
:attr:`grid` above. If any configurations matches the values specified in the
"key" dict, the values specified in the "values" dict will be searched instead.

Note, if arguments are commented out above but appear in the conditions, the
condition will be ignored.
"""

####################################
### DO NOT CHANGE THE CODE BELOW ###
####################################
### This code only has to be adapted if you are setting up this template for a
### new simulation script!

# Name of the script that should be executed by the hyperparameter search.
# Note, the working directory is set seperately by the hyperparameter search
# script, so don't include paths.
_SCRIPT_NAME = 'train_permutedMNIST.py'

# This file is expected to reside in the output folder of the simulation.
_SUMMARY_FILENAME = 'performance_overview.txt'

# These are the keywords that are supposed to be in the summary file.
# A summary file always has to include the keyword `finished`!.
_SUMMARY_KEYWORDS = [
    # Track all performance measures with respect to the best mean accuracy.
    'acc_after_list',
    'acc_during_list',
    'acc_after_mean',
    'acc_during_mean',
    'overall_task_infer_accuracy_list',
    'acc_task_infer_mean',
    'num_train_iter',
    'num_weights_class_net',
    'num_weights_rp_net',
    'num_weights_rp_hyper_net',
    'num_weights_class_hyper_net',
    'compression_ratio_class',
    'compression_ratio_rp',

    'finished'
]

# The name of the command-line argument that determines the output folder
# of the simulation.
_OUT_ARG = 'out_dir'

def _get_performance_summary(out_dir, cmd_ident):
    """See docstring of method
    :func:`hpsearch.hpsearch._get_performance_summary`.

    You only need to implement this function, if the default parser in module
    :func:`hpsearch.hpsearch` is not sufficient for your purposes.

    In case you would like to use a custom parser, you have to set the
    attribute :attr:`_SUMMARY_PARSER_HANDLER` correctly.
    """
    pass

# In case you need a more elaborate parser than the default one define by the
# function :func:`hpsearch.hpsearch._get_performance_summary`, you can pass a
# function handle to this attribute.
# Value `None` results in the usage of the default parser.
_SUMMARY_PARSER_HANDLE = None # Default parser is used.
#_SUMMARY_PARSER_HANDLE = _get_performance_summary # Custom parser is used.

def _performance_criteria(summary_dict, performance_criteria):
    """Evaluate whether a run meets a given performance criteria.

    This function is needed to decide whether the output directory of a run is
    deleted or kept.

    Args:
        summary_dict: The performance summary dictionary as returned by
            :attr:`_SUMMARY_PARSER_HANDLE`.
        performance_criteria (float): The performance criteria. E.g., see
            command-line option `performance_criteria` of script
            :mod:`hpsearch.hpsearch_postprocessing`.

    Returns:
        bool: If :code:`True`, the result folder will be kept as the performance
        criteria is assumed to be met.
    """
    ### Example:
    # return summary_dict['performance_measure1'] > performance_criteria

    raise NotImplementedError('TODO implement')

# A function handle, that is used to evaluate the performance of a run.
_PERFORMANCE_EVAL_HANDLE = None
#_PERFORMANCE_EVAL_HANDLE = _performance_criteria

# A key that must appear in the `_SUMMARY_KEYWORDS` list. If `None`, the first
# entry in this list will be selected.
# The CSV file will be sorted based on this keyword. See also attribute
# `_PERFORMANCE_SORT_ASC`.
_PERFORMANCE_KEY = None
assert(_PERFORMANCE_KEY is None or _PERFORMANCE_KEY in _SUMMARY_KEYWORDS)
# Whether the CSV should be sorted ascending or descending based on the
# `_PERFORMANCE_KEY`.
_PERFORMANCE_SORT_ASC = False

# FIXME: This attribute will vanish in future releases.
# This attribute is only required by the `hpsearch_postprocessing` script.
# A function handle to the argument parser function used by the simulation
# script. The function handle should expect the list of command line options
# as only parameter.
# Example:
# >>> from classifier.imagenet import train_args as targs
# >>> f = lambda argv : targs.parse_cmd_arguments(mode='cl_ilsvrc_cub',
# ...                                             argv=argv)
# >>> _ARGPARSE_HANDLE = f
_ARGPARSE_HANDLE = None

if __name__ == '__main__':
    pass


