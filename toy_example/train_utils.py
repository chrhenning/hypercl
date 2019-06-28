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
@title           :toy_example/train_utils.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :04/30/2019
@version         :1.0
@python_version  :3.6.8

A collection of helper functions to keep other scripts clean. The main purpose
is to collect the description of all command-line parameters.
"""
import torch
import torch.nn.functional as F
import tensorboardX
from tensorboardX import SummaryWriter
import argparse
import warnings
from datetime import datetime
import numpy as np
import random
import os
import shutil
import pickle
import itertools

from toy_example.regression1d_data import ToyRegression
import toy_example.gaussian_mixture_data as gmm_data
from toy_example.main_model import MainNetwork
from toy_example.hyper_model import HyperNetwork
from toy_example.task_recognition_model import RecognitionNet

import utils.misc as misc

def parse_cmd_arguments(mode='train_regression', default=False, argv=None):
    """Parse command-line arguments.

    Args:
        mode: For what script should the parser assemble the set of command-line
            parameters? Options:
                - "train_regression": Parser contains the set of parameters to
                  train a regression model for CL with 1D functions.
                - "train_mt_regression": Parser contains the set of parameters
                  to train a regression model for multi-task learning with 1D
                  functions.
        default (optional): If True, command-line arguments will be ignored and
            only the default values will be parsed.
        argv (optional): If provided, it will be treated as a list of command-
            line argument that is passed to the parser in place of sys.argv.

    Returns:
        The Namespace object containing argument names and values.
    """
    if mode == 'train_regression':
        description='Continual Learning - Toy Regression'
    elif mode == 'train_mt_regression':
        description='Multi-task Learning - Toy Regression'
    elif mode == 'train_ewc_regression':
        description='Continual Learning - EWC - Toy Regression'
    else:
        raise Exception('Mode "%s" unknown.' % (mode))
    parser = argparse.ArgumentParser(description=description)

    ### General training options
    parser.add_argument('--n_iter', type=int, default=4001,
                        help='Number of training iterations per task.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size.')
    parser.add_argument('--lr_hyper', type=float, default=1e-2,
                        help='Learning rate of the hyper network (or main ' +
                             'network if no hypernetwork is used).')
    parser.add_argument('--normal_init', action='store_true',
                        help='Use weight initialization from a zero-mean ' +
                             'normal with std defined by the argument ' +
                             '\'std_normal_init\'. Otherwise, Xavier ' +
                             'initialization is used. Biases are ' +
                             'initialized to zero.')
    parser.add_argument('--std_normal_init', type=float, default=0.02,
                        help='If normal initialization is used, this will ' +
                             'be the standard deviation used.')

    ### Dataset
    parser.add_argument('--dataset', type=int, default=0,
                        help='The dataloader to be used. Note, this option ' +
                             'does not define the precise task setting, only ' +
                             'the coarse structure of each task. The ' +
                             'following options are available: \n' +
                             '0: 1D function regression. \n' +
                             '1: Regression with inputs from tasks are drawn ' +
                             'from 2D Gaussian distributions.')

    ### Mainnet options
    parser.add_argument('--main_arch', type=str, default='10,10',
                        help='A string of comma-separated integers, each ' +
                             'denoting the size of a hidden layer of the ' +
                             'main network.')
    parser.add_argument('--main_act', type=str, default='sigmoid',
                        help='Activation function used in the main network.' +
                             'See option "hnet_act" for a list of options.')

    ### Multihead setting.
    parser.add_argument('--multi_head', action='store_true',
                        help='Use a multihead setting, where each task has ' +
                             'its own output head.')

    ### Miscellaneous options
    parser.add_argument('--val_iter', type=int, default=500,
                        help='How often the validation should be performed ' +
                             'during training.')
    parser.add_argument('--out_dir', type=str, default='./out/run_' +
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                        help='Where to store the outputs of this simulation.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Flag to disable GPU usage.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--data_random_seed', type=int, default=42,
                        help='The data is randomly generated at every run. ' +
                             'This seed ensures that the randomness during ' +
                             'data generation is decoupled from the training ' +
                             'randomness.')

    # If needed, add additional parameters.
    if mode == 'train_regression':
        parser = _hnet_arguments(parser)
        parser = _cl_arguments_general(parser)
        parser = _cl_arguments_ours(parser)
        parser = _cl_arguments_ewc(parser)
        parser = _ae_arguments(parser)
    elif mode == 'train_ewc_regression':
        parser = _cl_arguments_general(parser)
        parser = _cl_arguments_ewc(parser)
    elif mode == 'train_mt_regression':
        parser = _hnet_arguments(parser)
        parser = _mt_arguments(parser)
        parser = _ae_arguments(parser)

    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []
    config = parser.parse_args(args=args)

    ### Check argument values!
    assert(config.dataset in range(2))
    if mode == 'train_regression' or mode =='train_ewc_regression':
        if config.train_from_scratch and config.beta > 0:
            raise ValueError('"beta" should be 0 when training from scratch.')
        if mode == 'train_regression':
            assert(config.reg in range(4))
            assert(not config.ewc_weight_importance or config.reg == 0)
            if config.ewc_weight_importance:
                # See docstring of method "_cl_arguments_ewc" for an exlanation.
                assert(not config.online_ewc)

            if config.masked_reg:
                if not config.multi_head:
                    raise ValueError('Weights in the regularizer can only be ' +
                                     'masked when using a multi-head setup.')
                if config.reg not in range(3):
                    raise ValueError('Weights in the regularizer can only be ' +
                                     'masked when using regularizer 0, 1 or 2.')
                if config.reg in [1, 2]:
                    raise NotImplementedError('Reg masking not yet ' +
                        'implemented for chosen regularizer.')

                if config.plastic_prev_tembs and config.reg != 0:
                    raise ValueError('"plastic_prev_tembs" may only be ' +
                                     'enabled if "reg=0".')

    return config

def _cl_arguments_general(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    arguments to the parser that are specific to a continual learning setup.

    Args:
        parser: Object of class ArgumentParser.

    Returns:
        parser: CL arguments have been added.
    """
    ### Continual Learning options
    parser.add_argument('--beta', type=float, default=0.005,
                        help='Trade-off for the CL regularizer.')

    ### Miscellaneous options
    parser.add_argument('--train_from_scratch', action='store_true',
                        help='If set, all networks are recreated after ' +
                             'training on each task. Hence, training starts ' +
                             'from scratch.')
    return parser

def _cl_arguments_ours(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    arguments to the parser that are specific to our continual learning setup.

    This method expects that "_cl_arguments_general" has been called.

    Args:
        parser: Object of class ArgumentParser.

    Returns:
        parser: CL arguments have been added.
    """
    ### Continual Learning options
    parser.add_argument('--alpha', type=float, default=1,
                        help='Trade-off for the L2 reg of the weight change ' +
                             'when using the proximal algorithm.')
    parser.add_argument('--reg', type=int, default=0,
                        help='The regularizer to use to encourage continual ' +
                             'learning: \n' +
                             '0 - In this case, fixed targets are used in ' +
                             'the regularizer. These targets are simply the ' +
                             'hypernet outputs for previous task embeddings ' +
                             'before starting to learn the new task. \n' +
                             '1 - A simple regularizer is added to the task-' +
                             'specific loss that enforces a constant input-' +
                             'output mapping for previous tasks. See ' +
                             'method "calc_value_preserving_reg" of class ' +
                             '"HyperNetwork". \n' +
                             '2 - Same as 1, but a linearization via a first-' +
                             'order Taylor approximation has been applied. \n' +
                             '3 - Use EWC to avoid catastrophic forgetting ' +
                             'in the hypernetwork.')
    parser.add_argument('--use_proximal_alg', action='store_true',
                        help='Proximal algorithm. In this case, the ' +
                             'optimal weight change is searched for via ' +
                             'optimization rather than the actual weights.' +
                             'Note, in this case the options ' +
                             '"use_sgd_change" and "backprop_dt" have no ' +
                             'effect.')
    parser.add_argument('--n_steps', type=int, default=5,
                        help='When using the proximal algorithm, this ' +
                             'option decides the number of optimization ' +
                             'steps for seeking the next weight change.')
    parser.add_argument('--use_sgd_change', action='store_true',
                        help='This argument decides how delta theta (the ' +
                             'difference of the hypernet weights when taking ' +
                             'a step in optimizing the task-specific loss) ' +
                             'is computed. Note, delta theta is needed to ' +
                             'compute the CL regularizer. If this option is ' +
                             'True, then we approximate delta theta by its ' +
                             'SGD version: - alpha * grad, where alpha ' +
                             'represents the learning rate. This version is ' +
                             'computationally cheaper.')
    parser.add_argument('--backprop_dt', action='store_true',
                        help='Allow backpropagation through delta theta in ' +
                             'the regularizer.')
    parser.add_argument('--masked_reg', action='store_true',
                        help='Whether only used output weights should be ' +
                             'regularized in a multi-head setting. ' +
                             'Note, this only affects regularizers 0, 1 and 2.')
    parser.add_argument('--plastic_prev_tembs', action='store_true',
                        help='Allow the adaptation of previous task ' +
                             'embeddings. Note, by default we leave them ' +
                             'constant after learning the corresponding ' +
                             'task. However, allowing them to change when '+
                             'learning new tasks (while keeping their ' +
                             'targets fixed) should give more capacity and ' +
                             'flexibilty to the hypernet with no obvious ' +
                             'drawbacks. This option may only be enabled ' +
                             'when choosing "reg=0".')

    # TODO implement for reg 1 and 2.
    parser.add_argument('--ewc_weight_importance', action='store_true',
                        help='Can only be used with "reg=0". If enabled, ' +
                             'then the squarred error between hypernet ' +
                             'outputs and targets is regularized based on ' +
                             'weight importance. We use the empirical Fisher ' +
                             'estimate as importance measure for weights of ' +
                             'the main network. ' +
                             'Note, we don\'t allow the usage of online EWC.')

    return parser

def _cl_arguments_ewc(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    arguments to the parser that are specific to the EWC continual learning
    setup.

    This method expects that "_cl_arguments_general" has been called.

    Note, the online EWC algorithm doesn't make sense in a setup where the
    Fisher is estimated based on parameters that are outputted by a
    hypernetwork, as the Bayesian view does not apply here.

    For instance:
        W1 = h(c1, theta)
        ...
        WK = h(cK, theta)
    If tasks c1, ..., cK are learned in a continual setting, then the Bayesian
    view may apply to theta. But there is no need to view Wi as a prior for
    W{i+1} as these are potentially arbitrary outputs of the hypernetwork.
    Hence, we do not allow the online EWC algorithm, when calculating a Fisher
    estimate based on Wi, which is the output of a hypernetwork.

    Args:
        parser: Object of class ArgumentParser.

    Returns:
        parser: CL arguments have been added.
    """
    ### Continual Learning options
    parser.add_argument('--online_ewc', action='store_true',
                        help='Use online EWC algorithm (only applied if EWC is ' +
                             'used).')
    parser.add_argument('--gamma', type=float, default=1.,
                        help='Decay rate when using online EWC algorithm.')
    parser.add_argument('--n_fisher', type=int, default=-1,
                        help='Number of training samples to be used for the ' +
                             'estimation of the diagonal Fisher elements. If ' +
                             '"-1", all training samples are used.')
    return parser

def _ae_arguments(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    arguments to the parser that are specific to using a task recognition
    network.

    Args:
        parser: Object of class ArgumentParser.

    Returns:
        parser: AE arguments have been added.
    """
        ### Recognition network.
    parser.add_argument('--use_task_detection', action='store_true',
                        help='Enable the usage of a recognition model that ' +
                             'is trained in parallel and used for task ' +
                             'detection during inference.')
    parser.add_argument('--n_iter_ae', type=int, default=1001,
                        help='Number of training iterations per task of ' +
                             'the recognition model.')
    parser.add_argument('--lr_ae', type=float, default=0.001,
                        help='Learning rate of the recognition model.')
    parser.add_argument('--ae_arch', type=str, default='10,10',
                        help='A string of comma-separated integers, each ' +
                             'denoting the size of a hidden layer of the ' +
                             'encoder of the recognition model. The decoder ' +
                             'its architecture is mirrored. ' +
                             'Note, that the multitask baselines don\'t have ' +
                             'a decoder in the recognition model.')
    parser.add_argument('--ae_act', type=str, default='relu',
                        help='Activation function used in the recognition ' +
                             'model. See option "hnet_act" for a list of ' +
                             'options.')
    parser.add_argument('--ae_dim_z', type=int, default=8,
                        help='The dimensionality of the latent space z of ' +
                             'the recognition model.')
    parser.add_argument('--ae_beta_ce', type=float, default=10.,
                        help='The trade-off parameter for the cross-entropy ' +
                             'term in the loss of the recognition network.')
    parser.add_argument('--ae_beta_pm', type=float, default=1.,
                        help='The trade-off parameter for the prior-matching ' +
                             'term in the loss of the recognition network.')
    return parser

def _mt_arguments(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    arguments to the parser that are specific to the multi-task learning setup.

    Args:
        parser: Object of class ArgumentParser.

    Returns:
        parser: Multi-task arguments have been added.
    """
    parser.add_argument('--method', type=int, default=1,
                        help='Which multi-task setup should be trained: \n' +
                             '0 - In this case, only a main network with ' +
                             'trainable weights is generated to be trained ' +
                             'on all data at once. \n' +
                             '1 - In this case, the setup is comparable to ' +
                             'the continual learning setup, such that there ' +
                             'is a learned task embedding for each task. \n' +
                             '2 - In this case, there is only one task ' +
                             'embedding for all tasks, such that the ' +
                             'hypernetwork has to find an output that solves ' +
                             'all tasks at once.')

    return parser

def _hnet_arguments(parser):
    """This is a helper method of the method parse_cmd_arguments to add
    arguments to the parser that are specific to the usage of a hypernetwork.

    Args:
        parser: Object of class ArgumentParser.

    Returns:
        parser: Hypernet arguments have been added.
    """
    ### Hypernet options
    parser.add_argument('--hnet_arch', type=str, default='10,10',
                        help='A string of comma-separated integers, each ' +
                             'denoting the size of a hidden layer of the ' +
                             'hypernetwork.')
    parser.add_argument('--hnet_act', type=str, default='sigmoid',
                        help='Activation function used in the hypernetwork.' +
                             'The following options are available: ' +
                             '"linear": No activation function is used, ' +
                             '"sigmoid", "relu", "elu".')
    parser.add_argument('--emb_size', type=int, default=2,
                        help='Size of the task embedding space (input to ' +
                             'hypernet).')
    parser.add_argument('--std_normal_temb', type=float, default=1.,
                        help='Std when initializing task embeddings.')
    return parser

def _setup_environment(config):
    """Setup the general environment for training. This incorporates:
        * making computation deterministic
        * creating the output folder
        * selecting the torch device
        * creating the Tensorboard writer

    Args:
        config: Command-line arguments.

    Returns:
        device: Torch device to be used.
        writer: Tensorboard writer. Note, you still have to close the writer
            manually!
    """

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Output folder.
    if os.path.exists(config.out_dir):
        response = input('The output folder %s already exists. ' % \
                         (config.out_dir) + \
                         'Do you want us to delete it? [y/n]')
        if response != 'y':
            raise Exception('Could not delete output folder!')
        shutil.rmtree(config.out_dir)

        os.makedirs(config.out_dir)
        print("Created output folder %s." % (config.out_dir))

    else:
        os.makedirs(config.out_dir)
        print("Created output folder %s." % (config.out_dir))

    # Save user configs to ensure reproducibility of this experiment.
    with open(os.path.join(config.out_dir, 'config.pickle'), 'wb') as f:
        pickle.dump(config, f)

    ### Select torch device.
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using cuda: ' + str(use_cuda))

    ### Initialize summary writer.
    # Flushes every 120 secs by default.
    # DELETEME Ensure downwards compatibility.
    if not hasattr(tensorboardX, '__version__'):
        writer = SummaryWriter(log_dir=os.path.join(config.out_dir, 'summary'))
    else:
        writer = SummaryWriter(logdir=os.path.join(config.out_dir, 'summary'))

    return device, writer

def _generate_tasks(config):
    """Generate a set of user defined tasks.
    
    Args:
        config: Command-line arguments.

    Returns:
        data_handlers: A list of data handlers.
        num_tasks: Number of generated tasks.
    """
    if config.dataset == 0: # 1D function regression.
        return _generate_1d_tasks(config)
    elif config.dataset == 1: # Regression with GMM inputs.
        return _generate_gmm_tasks(config)
    else:
        raise ValueError('Dataset %d unknown!' % config.dataset)

def _generate_1d_tasks(config):
    """Generate a set of tasks for 1D regression.

    Args:
        See docstring of method "_generate_tasks".

    Returns:
        data_handlers: A data handler for each task (instance of class
            "ToyRegression").
        num_tasks: Number of generated tasks.
    """
    # FIXME task generation currently not controlled by user via command-line.
    if False:  # Set of random polynomials.
        num_tasks = 20
        x_domains = [[-10, 10]] * num_tasks

        # Disjoint x domains.
        #tmp = np.linspace(-10, 10, num_tasks+1)
        #x_domains = list(zip(tmp[:-1], tmp[1:]))

        max_degree = 6
        pcoeffs = np.random.uniform(-1, 1, size=(num_tasks, max_degree+1))

        map_funcs = []
        for i in range(num_tasks):
            d = np.random.randint(0, max_degree)
            # Ignore highest degrees.
            c = pcoeffs[i, d:]

            # Decrease the magnitute of higher order coefficients.
            f = .5
            for j in range(c.size-1, -1, -1):
                c[j] *= f
                f *= f

            # We want the border points of all polynomials to not exceed a
            # certain absolute magnitude.
            bp = np.polyval(c, x_domains[i])
            s = np.max(np.abs(bp)) + 1e-5
            c = c / s * 10.

            map_funcs.append(lambda x, c=c : np.polyval(c, x))

        std = .1

    else: # Manually selected tasks.
        """
        tmp = np.linspace(-15, 15, num_tasks + 1)
        x_domains = list(zip(tmp[:-1], tmp[1:]))
        map_funcs = [lambda x: 2. * (x+10.),
                     lambda x: np.power(x, 2) * 2./2.5 - 10,
                     lambda x: np.power(x-10., 3) * 1./12.5]
        std = 1.
        """

        """
        map_funcs = [lambda x : 0.1 * x, lambda x : np.power(x, 2) * 1e-2,
                     lambda x : np.power(x, 3) * 1e-3]
        num_tasks = len(map_funcs)
        x_domains = [[-10, 10]] * num_tasks
        std = .1
        """

        map_funcs = [lambda x : (x+3.), 
                     lambda x : 2. * np.power(x, 2) - 1,
                     lambda x : np.power(x-3., 3)]
        num_tasks = len(map_funcs)
        x_domains = [[-4,-2], [-1,1], [2,4]]
        std = .05


        """
        map_funcs = [lambda x : (x+30.),
                     lambda x : .2 * np.power(x, 2) - 10,
                     lambda x : 1e-2 * np.power(x-30., 3)]
        num_tasks = len(map_funcs)
        x_domains = [[-40,-20], [-10,10], [20,40]]
        std = .5
        """


    dhandlers = []
    for i in range(num_tasks):
        print('Generating %d-th task.' % (i))
        dhandlers.append(ToyRegression(train_inter=x_domains[i],
            num_train=100, test_inter=x_domains[i], num_test=50,
            map_function=map_funcs[i], std=std, rseed=config.data_random_seed))
        dhandlers[-1].plot_dataset()

    return dhandlers, num_tasks

def _generate_gmm_tasks(config):
    """Generate a set of regression tasks with inputs drawn from a Gaussian
    mixture model.

    Args:
        See docstring of method "_generate_tasks".

    Returns:
        data_handlers: A data handler for each task (instance of class
            "GaussianData").
        num_tasks: Number of generated tasks.
    """
    # FIXME task generation currently not controlled by user via command-line.

    if True:
        means = [np.array([i, j]) for i, j in
                 itertools.product(range(-4, 5, 4), range(-4, 5, 4))]
        std = .5
    else:
        means = gmm_data.DEFAULT_MEANS
        # For density estimation, the variances shouldn't be too small.
        std = 0.2

    covs = [std**2 * np.eye(len(mean)) for mean in means]
    dhandlers = gmm_data.get_gmm_taks(means=means, covs=covs, num_train=1000,
        num_test=50, rseed=config.data_random_seed)
    num_tasks = len(dhandlers)

    #for i in range(num_tasks):
    #    print('Task %d:' % (i))
    #    dhandlers[i].plot_dataset()
    gmm_data.GaussianData.plot_datasets(dhandlers)

    return dhandlers, num_tasks

def _generate_networks(config, data_handlers, device, create_hnet=True,
                       create_rnet=False, no_replay=False):
    """Create the main-net, hypernetwork and recognition network.

    Args:
        config: Command-line arguments.
        data_handlers: List of data handlers, one for each task. Needed to
            extract the number of inputs/outputs of the main network. And to
            infer the number of tasks.
        device: Torch device.
        create_hnet: Whether a hypernetwork should be constructed. If not, the
            main network will have trainable weights.
        create_rnet: Whether a task-recognition autoencoder should be created.
        no_replay: If the recognition network should be an instance of class
            MainModel rather than of class RecognitionNet (note, for multitask
            learning, no replay network is required).

    Returns:
        mnet: Main network instance.
        hnet: Hypernetwork instance. This return value is None if no
            hypernetwork should be constructed.
        rnet: RecognitionNet instance. This return value is None if no
            recognition network should be constructed.
    """
    num_tasks = len(data_handlers)

    n_x = data_handlers[0].in_shape[0]
    n_y = data_handlers[0].out_shape[0]
    if config.multi_head:
        n_y = n_y * num_tasks

    def str_to_act(act_str):
        if act_str == 'linear':
            act = None
        elif act_str == 'sigmoid':
            act = torch.nn.Sigmoid()
        elif act_str == 'relu':
            act = torch.nn.ReLU()
        elif act_str == 'elu':
            act = torch.nn.ELU()
        else:
            raise Exception('Activation function %s unknown.' % act_str)
        return act

    main_arch = misc.str_to_ints(config.main_arch)
    main_shapes = MainNetwork.weight_shapes(n_in=n_x, n_out=n_y,
                                            hidden_layers=main_arch)
    mnet = MainNetwork(main_shapes, activation_fn=str_to_act(config.main_act),
                       use_bias=True, no_weights=create_hnet).to(device)
    if create_hnet:
        hnet_arch = misc.str_to_ints(config.hnet_arch)
        hnet = HyperNetwork(main_shapes, num_tasks, layers=hnet_arch,
            te_dim=config.emb_size,
            activation_fn=str_to_act(config.hnet_act)).to(device)
        init_params = list(hnet.parameters())
    else:
        hnet = None
        init_params = list(mnet.parameters())

    if create_rnet:
        ae_arch = misc.str_to_ints(config.ae_arch)
        if no_replay:
            rnet_shapes = MainNetwork.weight_shapes(n_in=n_x, n_out=num_tasks,
                hidden_layers=ae_arch, use_bias=True)
            rnet = MainNetwork(rnet_shapes,
                activation_fn=str_to_act(config.ae_act), use_bias=True,
                no_weights=False, dropout_rate=-1,
                out_fn=lambda x : F.softmax(x, dim=1))
        else:
            rnet = RecognitionNet(n_x, num_tasks, dim_z=config.ae_dim_z,
                                  enc_layers=ae_arch,
                                  activation_fn=str_to_act(config.ae_act),
                                  use_bias=True).to(device)
        init_params += list(rnet.parameters())
    else:
        rnet = None

    ### Initialize network weights.
    for W in init_params:
        if W.ndimension() == 1: # Bias vector.
            torch.nn.init.constant_(W, 0)
        elif config.normal_init:
            torch.nn.init.normal_(W, mean=0, std=config.std_normal_init)
        else:
            torch.nn.init.xavier_uniform_(W)

    # The task embeddings are initialized differently.
    if create_hnet:
        for temb in hnet.get_task_embs():
            torch.nn.init.normal_(temb, mean=0., std=config.std_normal_temb)

    return mnet, hnet, rnet

if __name__ == '__main__':
    pass