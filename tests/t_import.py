import math
print(math.pi)
import sys as msys
print(msys.platform)
import os.path
print(os.path.sep)
from builtins import repr as re
print(repr is re)


# Regression: `from m import *` used to emit `module.*` (a syntax
# error). Now it pulls every non-underscore name in dir() (or
# __all__) into globals().
from math import *
print(round(sqrt(81), 3))
