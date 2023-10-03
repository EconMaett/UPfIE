########################################################################
# Project X:
# "The Ultimate Question of Life, the Universe, and Everything"
# Project Collaborators: Mr. X, Mrs. Y
#
# Python Script "ultimate-calcs"
# by: F Heiss
# Date of this version: February 18, 2019
########################################################################
# external modules:
import numpy as np
import datetime as dt

# create a time stamp:
ts = dt.datetime.now()

# print to logfile.txt ('w' resets the logfile before writing output)
# in the provided path (make sure that the folder structure
# you may provide already exists):
print(f'This is a log file from: \n{ts}\n',
      file=open('Pyout/19/logfile.txt', 'w'))

# the first calculation using the function "square root" from numpy:
result1 = np.sqrt(1764)

# print to logfile.txt but with keeping the previous results ('a'):
print(f'result1: {result1}\n',
      file=open('Pyout/19/logfile.txt', 'a'))

# the second calculation reverses the first one:
result2 = result1 ** 2

# print to logfile.txt but with keeping the previous results ('a'):
print(f'result2: {result2}',
      file=open('Pyout/19/logfile.txt', 'a'))
