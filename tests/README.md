# Test cases for the code implemented in this repository

Note, the subfolders in this directory should have the same structure as in the main repository (parent folder), except for the prefix `test_`, which is necessary as no two folders in this repo may have the same name (see [contribution guide](../CONTRIBUTING.md)).

```console
$ python3 -m unittest discover -t ..
```

Note, test case developers are urged to only allow command-line printing for their test cases if the verbosity level is set to `2` (see method [unittest_verbosity()](test_utils.py)), i.e., the option `-v` is passed:

```console
$ python3 -m unittest discover -t .. -v
```
