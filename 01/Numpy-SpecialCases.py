import numpy as np

# array of integers defined by the arguments start, end and sequence length:
sequence = np.linspace(0, 2, num=11)
print(f'sequence: \n{sequence}\n')

# sequence of integers starting at 0, ending at 5-1:
sequence_int = np.arange(5)
print(f'sequence_int: \n{sequence_int}\n')

# initialize array with each element set to zero:
zero_array = np.zeros((4, 3))
print(f'zero_array: \n{zero_array}\n')

# initialize array with each element set to one:
one_array = np.ones((2, 5))
print(f'one_array: \n{one_array}\n')

# uninitialized array (filled with arbitrary nonsense elements):
empty_array = np.empty((2, 3))
print(f'empty_array: \n{empty_array}\n')
