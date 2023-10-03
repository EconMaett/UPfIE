import numpy as np

# define arrays in numpy:
testarray1D = np.array([1, 5, 41.3, 2.0])
print(f'type(testarray1D): {type(testarray1D)}\n')

testarray2D = np.array([[4, 9, 8, 3],
                        [2, 6, 3, 2],
                        [1, 1, 7, 4]])

# get dimensions of testarray2D:
dim = testarray2D.shape
print(f'dim: {dim}\n')

# access elements by indices:
third_elem = testarray1D[2]
print(f'third_elem: {third_elem}\n')

second_third_elem = testarray2D[1, 2]  # element in 2nd row and 3rd column
print(f'second_third_elem: {second_third_elem}\n')

second_to_third_col = testarray2D[:, 1:3]  # each row in the 2nd and 3rd column
print(f'second_to_third_col: \n{second_to_third_col}\n')

# access elements by lists:
first_third_elem = testarray1D[[0, 2]]
print(f'first_third_elem: {first_third_elem}\n')

# same with Boolean lists:
first_third_elem2 = testarray1D[[True, False, True, False]]
print(f'first_third_elem2: {first_third_elem2}\n')

k = np.array([[True, False, False, False],
              [False, False, True, False],
              [True, False, True, False]])
elem_by_index = testarray2D[k]  # 1st elem in 1st row, 3rd elem in 2nd row...
print(f'elem_by_index: {elem_by_index}\n')
