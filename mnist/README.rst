

MNIST experiments for continual learning with deterministic hypernetworks
***************************************************************************

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

In this subpackage, we investigate two classic MNIST continual learning experiments.
We strictly follow the experimental setup proposed in the `Three scenarios for continual learning paper <https://arxiv.org/abs/1904.07734>`__.
The implementation from the same authors for many methods we compare to can be found in 
the `Three scenarios for continual learning repo <https://github.com/GMvandeVen/continual-learning>`__.



MNIST continual learning experiments
------------------------------------

.. _mnist-readme-reference-label:

The two standard MNIST experiments are the following:

splitMNIST 
^^^^^^^^^^
In this experiment, every single task (max 5) consists of distinguishing between
two MNIST classes, i.e., digits. For example, the second task consists of images 
of twos and threes, in the fifth task we need to learn to differentiate eights 
and nines. To start a splitMNIST experiment, execute the following command:

.. code-block:: console

    $ python train_splitMNIST.py

permutedMNIST 
^^^^^^^^^^^^^
In this experiment, the first task is the standard MNIST classification task.
For every consecutive tasks, we randomly permute all input images by a 
fixed permutation per task and keep the labels fixed.
To start a permutedMNIST experiment, execute the following command:

.. code-block:: console

    $ python train_permutedMNIST.py

In our paper, we investigate very long task sequences. You can specify the 
number of tasks that you want to train on by setting ``--num_tasks`` e.g. execute

.. code-block:: console

    $ python train_permutedMNIST.py --num_tasks 100

if you want to train on 100 permutedMNIST sequentially (default set to 10).


Continual learning scenarios
----------------------------

Following the `Three scenarios for continual learning paper <https://arxiv.org/abs/1904.07734>`__,
we make the same distinguishment between three learning scenarios. Note, that 
throughout the paper we know when task changes occur during training i.e. task boundaries are
known.

CL scenario 1 
^^^^^^^^^^^^^
In this scenario, we assume that the learner is given the task id for every 
input during inference. 
During training, after a new task is given, we start training a new
output head and assume we know the number of classes per task.
Knowing the task id at inference time, we can choose the output head 
that was used during training for the given task. This cl scenario is activated 
by default.

CL scenario 2 
^^^^^^^^^^^^^
In this scenario, we assume that the number of classes per task stays constant 
throughout all tasks. Therefore, we train the same output head on all tasks but
different from cl scenario 1, we are not given the task id during inference. 
You can choose the learning scenario by setting ``--cl_scenario`` i.e. execute:

.. code-block:: console

    $ python train_splitMNIST.py --cl_scenario 2

CL scenario 3
^^^^^^^^^^^^^
Similar to CL scenario 1, in this learning scenario we train every task on a
separate output head but are not given the task id during inference.
Normally in this scenario, one computes the softmax overall 
output neurons of all tasks to make predictions (classes per task * number of tasks).
Again, you set this cl scenario by executing:

.. code-block:: console

    $ python train_splitMNIST.py --cl_scenario 3

In our paper, we investigate the possibility to infer the task id. This makes
it possible to narrow down the output neurons we compute the softmax over.
This `devide and conquer` principle seems to be especially powerful when the data 
distributions of the different tasks are easily seperable e.g. in permutedMNIST 
experiments. 


Our three different methods
---------------------------

We propose three different methods to show the wide applicability of hypernetworks
for different continual learning approaches.

Hypernetworks empowering a replay model (HNET+R)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay of (synthetic) data has proven itself to be a (the?) successful 
continual learning solution in all cl scenarios. In this method (chosen by default), 
we use a hypernetwork to empower sequential training of a decoder of 
a variational autoencoder (chosen by default, GAN not tested). 
With this replay model, we can then train a classifier on the current tasks and on the replay data from the VAE.
This method is very similar to the `Continual Learning with Deep Generative Replay paper <https://arxiv.org/abs/1705.08690>`__,
but instead of using the replay data to train itself to not forget old tasks, 
the replay model protects itself against catastrophic forgetting via a regularised
hypernetwork.  


Hypernetworks empowering a task inference model (HNET+TIR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For cl scenario 1, this is the most straight forward use of hypernetworks for 
continual learning. Here, we simply train the hypernetwork consecutively on 
different tasks and protect learned models by our simple regularisation.
During inference, we know the task id and can choose the corresponding embedding 
and output head. This has proven itself very successful, especially in long 
task sequences. To exploit this, for cl scenario 2 and 3, we propose to train   
a task inference classifier with a replay model, similar to HNET+R. 
Training this task id inferring classifier is similar to class incremental training, 
where data of a single class now is data from a whole task. 

Use this method, termed HNET+TIR, by setting ``--infer_task_id``, i.e. by executing

.. code-block:: console

    $ python train_splitMNIST.py --infer_task_id --cl_scenario i

where :math:`i \in \{1,2,3\}`.


Task inference through entropy (HNET+ENT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method is inspired by the success of the task inference model (HNET+TIR) i.e. 
the divide and conquer approach. Here, we try to infer the task id by choosing 
the model/output head with the lowest entropy of the corresponding prediction. Therefore, we 
iterate of the learned embeddings (and output heads in cl scenario 3),
compute the entropy of the predictions and choose the one with the lowest entropy.

This method is only supported when ``--infer_task_id`` is set. Therefore, use this 
method by setting ``--infer_with_entropy``, i.e. by executing

.. code-block:: console

    $ python train_splitMNIST.py --infer_task_id --infer_with_entropy --cl_scenario i


where :math:`i \in \{2,3\}`. For cl scenario 1 we do not need to infer the task id.


Additional notes
----------------

By default, all hyperparameters to reproduce the results in the paper are set.
To deactivate this behaviour, execute the scripts by additionally setting 
``--dont_set_default`` i.e. by executing:

.. code-block:: console

    $ python train_splitMNIST.py --dont_set_default


We did not look into Generative Adversarial Networks empowered by hypernetworks
in detail. To reproduce the images shown in the paper, follow the 
training details reported in the appendix.

Related work
------------

We used the code from `van de Ven et al. <https://github.com/GMvandeVen/continual-learning/>`__ to perform experiments on related work methods. See their `paper <https://arxiv.org/abs/1904.07734>`__ for more infos on the setup.

For the PermutedMNIST-100 experiments, we ran their code with 5 different random seeds using the following command for `Online EWC`:

.. code-block:: console

    $ ./main.py --ewc --online --lambda=100 --gamma=1 --experiment permMNIST --scenario task --tasks 100 --fc-units=1000 --lr=0.0001 --iters 5000 --seed $i

for `Synaptic Intelligence`

.. code-block:: console

    $ ./main.py --si --c=0.1 --experiment permMNIST --scenario task --tasks 100 --fc-units=1000 --lr=0.0001 --iters 5000 --seed $i

and for `DGR+distill`

.. code-block:: console

    $ ./main.py --replay=generative --distill  --experiment permMNIST --scenario task --tasks 100 --fc-units=1000 --lr=0.0001 --iters 5000 --seed $i

To assess the susceptibility of `Online EWC` on the parameter ``lambda``, we ran the following command for the ``lambda`` values ``[1, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000]`` and 5 different random seeds each

.. code-block:: console

    $ python3 main.py --ewc --online --lambda=$i --gamma=1 --experiment permMNIST --scenario task --tasks 100 --fc-units=1000 --lr=0.0001 --iters=5000 --seed=$j
