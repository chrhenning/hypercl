import os
import sys

curr_dir = os.path.basename(os.path.abspath(os.curdir))
# We want to ensure that people can run scripts from the top-level directory of
# this package (in which case no further action is required) and from its
# subfolders. For instance:
# From directory PACKAGE/toy_example:
#   $ python3 train.py
# From directory PACKAGE/
#   $ pyhton3 -m toy_example.train
# Both ways of calling a script should be possible. If a script is called from
# a subfolder, it is necessary to add the parent dir to the path.
# Note, the user might rename the folder PACKAGE, which is why we can't rely on
# it.
base_dir = os.path.abspath('..')
if curr_dir == 'toy_example' and base_dir != sys.path[0]:
    sys.path.insert(0, base_dir)

# Initialize plotting environment.
from utils import misc
misc.configure_matplotlib_params()
