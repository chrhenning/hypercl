import os
import sys

curr_dir = os.path.basename(os.path.abspath(os.curdir))
# See __init__.py in folder "toy_example" for an explanation.
trgt_dir = os.path.abspath('..')
if curr_dir == 'tests' and trgt_dir != sys.path[0]:
    sys.path.insert(0, trgt_dir)
    
