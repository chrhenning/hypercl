CIFAR-10/100 experiments for continual learning with deterministic hypernetworks
================================================================================

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

**Note**, the code in this package originated from the subpackage ``classifier``, which is deprecated. The code was cleaned up and we improved readibility and consistency. However, the code is therfore not sufficiently tested and should be used with extra caution.

In this subpackage we investigate an experiment proposed in the `Synaptic Intelligence (SI) paper <https://arxiv.org/abs/1703.04200>`__, which consists of 6 tasks. The first task is to learn CIFAR-10. The remaining 5 tasks consist of splits from CIFAR-100. Each split contains 10 classes, such that the first 50 classes of CIFAR-100 will be learned (thus, only half of the training set is used). Since both datasets contain the same amount of training samples, the final 5 tasks have only a tenth of the number of training samples available compared to task 1. Thus, this benchmark is particularly suitable to qualitatively measure forward transfer.

.. _cifar-readme-zenke-reference-label:

Experiments using the same network as Zenke et al.
--------------------------------------------------

In this experiment we reproduce the experimental conditions from the `Synaptic Intelligence (SI) paper <https://arxiv.org/abs/1703.04200>`_.

Our results with this architecture were obtained using the following command (for 5 different random seeds)

.. code-block:: console

    $ python3 train_zenke.py --use_adam --custom_network_init --disable_data_augmentation --cl_scenario=1


Slightly better results can be obtained using the extension ``--soft_targets``. The during accuracies can be significantly improved when using data augmentation. Though, as far as we know, Zenke et al. didn't use any data augmentation in their experiments. So we didn't explore this option.

To see the full set of available options type

.. code-block:: console

    $ python3 train_zenke.py --help

.. _cifar-readme-resnet-reference-label:

Resnet-32 experiments
---------------------

The Resnet-32 SplitCIFAR experiment can be reproduced using the following command (using 5 different random seeds)

.. code-block:: console

    $ python3 train_resnet.py --use_adam --custom_network_init --plateau_lr_scheduler --lambda_lr_scheduler

The `training from scratch` experiments can be performed adding the option ``--training_from_scratch`` and by disabling the regularizer via ``--beta=0``. The `fine-tuning` experiments can be performed by disabling the regularizer (``--beta=0``) and additionally by disabling the checkpointing of running statistics via ``--bn_no_stats_checkpointing``. Note, without this last option, the fine-tuned batchnorm weights will not be in sync with the checkpointed statistics anymore after new tasks have been learned and the accuracy of previous tasks will trivially fall to chance level.

To see the full set of available options type

.. code-block:: console

    $ python3 train_resnet.py --help

