import numpy as np
import sys

with open(sys.argv[1], 'r') as f:
  re = f.read().splitlines()
re = filter(lambda e: '#' in e, re)
for e in re:
  print e
