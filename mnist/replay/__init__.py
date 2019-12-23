import os
import sys

curr_dir = os.path.basename(os.path.abspath(os.curdir))
# See __init__.py in folder "toy_example" for an explanation.
if curr_dir == 'replay' and '../..' not in sys.path:
    sys.path.insert(0, '../..')