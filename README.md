# Continual Learning via Hypernetworks

A continual learning approach that has the flexibility to learn a dedicated set of parameters, fine-tuned for every task, that doesn't require an increase in the number of trainable weights and is robust against catastrophic forgetting.

For details on this approach please read [our paper](https://arxiv.org/abs/1906.00695).

## Toy Examples

Some toy regression problems can be explored in the folder [toy_example](toy_example). Please refer to the corresponding [documentation](toy_example/README.md). Example run:

```console
$ python3 -m toy_example.train --no_cuda
```

## Classification Problems

Classical to challenging classification benchmarks are in the folder [classifier](classifier). Please refer to the corresponding [documentation](classifier/README.md).

## Citation
Please cite our paper if you use this code in your research project.

```
@article{oshg2019hypercl,
  title={Continual learning with hypernetworks},
  author={von Oswald, Johannes and Henning, Christian and Sacramento, Jo{\~a}o and Grewe, Benjamin F},
  journal={arXiv preprint arXiv:1906.00695},
  year={2019}
}
```
