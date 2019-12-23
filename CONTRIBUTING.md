# Contributing to this repository

All contributions and improvements are very welcome and appreciated.

First of all (and **most importantly**), we only want to have code contributions that follow basic coding guidelines. Our coding is inspired by [these guidelines](https://google.github.io/styleguide/pyguide.html). Here are some basic remarks:

* Lines should be no longer than **80 characters**
* All names (even local variables) must reflect the meaning of its entity
* All methods/functions need a docstring describing its purpose, arguments and return values
* If possible, program object-oriented
* All changes should be well documented
* Whenever possible, do not copy code
* Commit messages have to be precise, comprehensive and complete
* Never push buggy code to the *master* repo; be careful to not break other parts of the repo (a good sanity check if always to double-check your changes by running the [test suite](tests/README.md), but be aware that our tests only cover a small fraction of our code base)
* Never commit any temporary or result files to the repository (output directories of simulations; compiled code or IDE-specific configs). Also, *never* commit any large files or datasets.
* Whenever you pull from the repo, carefully check all changes made by others, especially in all files that affect your code.
* If you push third-party code to the repository, make sure that all attributions are of **copyright-holders** are done correctly and the **license** information is properly incorporated (note, this also holds for code snippets from [stackoverflow](https://stackoverflow.blog/2009/06/25/attribution-required/)) 
* **Never assume that the user knows how to use your program or function.** Capture all possible failure modes and provide meaningful warning/error messages.

**Use common sense when coding!**

> The best programs are written so that computing machines can perform them quickly and so that human beings can understand them clearly. A programmer is ideally an essayist who works with traditional aesthetic and literary forms as well as mathematical concepts, to communicate the way that an algorithm works and to convince a reader that the results will be correct.
> - Donald Ervin Knuth, Selected Papers on Computer Science

Note, since the repository is changing all the time, code gets deprecated freqeuently. Therefore, please run your python scripts with the option `--Wall` every once in a while to fix all warnings (especially deprecation warnings).

## Documentation

We follow the [Google styleguide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) when writing docstrings. We use [sphinx](http://www.sphinx-doc.org/en/master/usage/installation.html) to build documentation for this project. The documentation source files are located in [docs](docs/).

When you add your implementation files (containing docstrings) to the documentation source files, **make sure the documentation compiles without warnings**.

See [this example](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google) on how to write Google-style docstrings in [sphinx](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).

**Please fix any mistakes and inconsistencies you spot in the documentation.**

## Folder structure

We decided to have the flexibility of being able to execute scripts from their relative directories as well as from the base directory. Unfortunately, this flexibility comes with some limitations, most notably, we need to manually modify the `path` variable whenever executing scripts from their relative directory. See [this file](toy_example/__init__.py) for more details. most importantly, **no two directories in this repo may have the same name**. Otherwise, the *import* system might get confused, which leads to runtime errors or (in the worst case) to undesired behavior.

Hence, **every subfolder needs a custom `__init__.py` file and every executable module needs to start with the following lines:**

```python
# Do not delete the following import for all executable scripts!
import __init__ # pylint: disable=unused-import
```

In addition, **all imports most be absolute, i.e., with respect to the base directory** of this repository.

## Command-line arguments

Command-line argument definitions should be reused whenever possible (see module [cli_args](utils/cli_args.py)). Their help-text should be comprehensive (and ideally contain the default value). **Flags should always have the action `store_true`**, such that their usage is intuitive. This flag may only be changed temporarily if a very prominent warning is placed in the code.

**Default values should never change**, especially if other parts of the repo are reusing the argument definition (though, you may capture cases in the definition of a command-line argument to realize different default values for different cases).

An example definition for command line-arguments can be found [here](cifar/train_args.py).

## General setup and checkpointing

For the general setup of scripts the method [setup_environment](utils/sim_utils.py) should be used (see the corresponding docstring). Note, that this method requires some general command-line arguments to be set (see method [miscellaneous_args](utils/cli_args.py)).

Ideally, networks are regularly checkpointed by using the methods provided in [torch_ckpts](utils/torch_ckpts.py). Scripts (especially for extensive simulations) should provide the possibility to load and continue from existing checkpoints (such that no computational ressources are wasted).

## Data loaders

A set of framework-agnostic dataloaders is implemented in the folder [data](data). All dataloaders inherit their functionality from the class [Dataset](data/dataset.py).

These data loaders are easy to use (and tested with PyTorch and Tensorflow). Most dataloaders simply load the whole dataset in working memory. Some specialized data loaders (e.g., those derived from [LargeImgDataset](data/large_img_dataset.py) also allow the retrieval of costumized data loaders, more suitable for the Deep Learning framework at use.

The data loaders in the folder [data](data) are only for common datasets. There are several other data loaders (for special datasets, such as synthetic ones) that also inherit from class [Dataset](data/dataset.py) sprinkled across the repository.

If you want to implement you own data handler, such inherit from the abstract class [Dataset](data/dataset.py) and specify the fields of the *dict* attribute `_data` (there is a short description of all its keys in the constructor of [Dataset](data/dataset.py)). Additionally, you have to overwrite the abstract methods `get_identifier` and `_plot_sample`.

## Main- and Hypernetworks

A *main network* is a network that solves an actual task (specified through a dataset). *Hypernetworks* are used to generate the parameters of a corresponding main network. Both network types have a dedicated interface. Main networks are located in the folder [mnets](./mnets) and have to implement the interface [mnet_interface](mnets/mnet_interface.py). Hypernetworks are located in the folder [hnets](./hnets) and have to implement the interface [hnet_interface](hnets/hnet_interface.py). 

**Note, the design of the [hnet_interface](hnets/hnet_interface.py) is not completed yet. Currently (7th Nov 2019), main networks and hypernetworks are sprinkled across the repository. Some main networks have already been integrated into the folder [mnets](./mnets). Once a network has been migrated and implements the new interface, then the old network should be deprecated.**

All old networks are implementing the following interface [module_wrappers](utils/module_wrappers.py).

## Plotting

If possible, plots should be produced *publication-ready*, meaning you should make an effort to make them look nice from the beginning. To ensure consistency, we defined some general guidelines for our plots in the methods `configure_matplotlib_params` and `get_colorbrewer2_colors` of the modules [misc](utils/misc.py) (e.g., colors, fontsizes, font, ...).

Optimally, you call the method `configure_matplotlib_params` in the corresponding [__init__.py](__init__.py) file of your subfolder to ensure the plotting environment is consistently configured across the repository (e.g., see [here](toy_example/__init__.py) for an example).
