# external modules:
import numpy as np
import datetime as dt
import sys

# make sure that the folder structure you may provide already exists:
sys.stdout = open('Pyout/19/logfile2.txt', 'w')

# create a time stamp:
ts = dt.datetime.now()

# print to logfile2.txt:
print(f'This is a log file from: \n{ts}\n')

# the first calculation using the function "square root" from numpy:
result1 = np.sqrt(1764)

# print to logfile2.txt:
print(f'result1: {result1}\n')

# the second calculation reverses the first one:
result2 = result1 ** 2

# print to logfile2.txt:
print(f'result2: {result2}')
