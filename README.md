# Continual Learning with Hypernetworks

A continual learning approach that has the flexibility to learn a dedicated set of parameters, fine-tuned for every task, that doesn't require an increase in the number of trainable weights and is robust against catastrophic forgetting.

For details on this approach please read [our paper](https://arxiv.org/abs/1906.00695). You can find our spotlight presentation [here](https://iclr.cc/virtual_2020/poster_SJgwNerKvB.html) and a more detailed introduction in [this talk](https://youtu.be/sFNAXF8H0IY?t=959). Experiments on continual learning with hypernetworks using sequential data and recurrent networks can be found in [this repository](https://github.com/mariacer/cl_in_rnns). Furthermore, [this repository](https://github.com/chrhenning/posterior_replay_cl) studies a probabilistic extension of the proposed CL algorithm.

If you are interested in working with **hypernetworks in PyTorch**, check out the package [hypnettorch](https://github.com/chrhenning/hypnettorch). The package also provides an [example implementation](https://hypnettorch.readthedocs.io/en/latest/examples.html#continual-learning-with-hypernetworks) of our method for task-incremental learning.

## Toy Examples

Some toy regression problems can be explored in the folder [toy_example](toy_example). Please refer to the corresponding [documentation](toy_example/README.md). Example run:

```console
$ python3 -m toy_example.train --no_cuda
```

## MNIST Experiments

You can find instructions on how to reproduce our MNIST experiments and on how to use the corresponding code in the subfolder [mnist](mnist).

## CIFAR Experiments

Please checkout the subfolder [cifar](cifar). You may use the script [cifar.train_zenke](cifar/train_zenke.py) to run experiments using the same network as [Zenke et al.](https://arxiv.org/abs/1703.04200) and the script [cifar.train_resnet](cifar/train_resnet.py) to run experiments with a [Resnet-32](https://arxiv.org/abs/1512.03385).

## Testing

All testing of implemented functionality is located in the subfolder [tests](tests) and documented [here](tests/README.md). To run all unit tests, execute:

```console
$ python3 -m unittest discover -s tests/ -t .
```

## Documentation

Please refer to the [README](docs/README.md) in the subfolder [docs](docs) for instructions on how to compile and open the documentation.

## Setup Python Environment

We use [conda](https://www.anaconda.com/) to manage Python environments. To create an environment that already fulfills all package requirements of this repository, simply execute

```console
$ conda env create -f environment.yml
$ conda activate hypercl_env
```

## Citation
Please cite our paper if you use this code in your research project.

```
@inproceedings{ohs2019hypercl,
title={Continual learning with hypernetworks},
author={Johannes von Oswald and Christian Henning and Benjamin F. Grewe and Jo{\~a}o Sacramento},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://arxiv.org/abs/1906.00695}
}
```
