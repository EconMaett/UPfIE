import numpy as np

# define an arrays in numpy:
mat1 = np.array([[4, 9, 8],
                 [2, 6, 3]])
mat2 = np.array([[1, 5, 2],
                 [6, 6, 0],
                 [4, 8, 3]])

# use a numpy function:
result1 = np.exp(mat1)
print(f'result1: \n{result1}\n')

result2 = mat1 + mat2[[0, 1]]  # same as np.add(mat1, mat2[[0, 1]])
print(f'result2: \n{result2}\n')

# use a method:
mat1_tr = mat1.transpose()
print(f'mat1_tr: \n{mat1_tr}\n')

# matrix algebra:
matprod = mat1.dot(mat2)  # same as  mat1 @ mat2
print(f'matprod: \n{matprod}\n')
