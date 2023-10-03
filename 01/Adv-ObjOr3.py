import numpy as np

# multiply these two matrices:
a = np.array([[3, 6, 1], [2, 7, 4]])
b = np.array([[1, 8, 6], [3, 5, 8], [1, 1, 2]])


# define your own class:
class myMatrices:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def mult(self):
        N = self.A.shape[0]  # number of rows in A
        K = self.B.shape[1]  # number of cols in B
        out = np.empty((N, K))  # initialize output
        for i in range(N):
            for j in range(K):
                out[i, j] = sum(self.A[i, :] * self.B[:, j])
        return out


# define a subclass:
class myMatNew(myMatrices):
    def getTotalElem(self):
        N = self.A.shape[0]  # number of rows in A
        K = self.B.shape[1]  # number of cols in B
        return N * K


# create an object of the subclass:
test = myMatNew(a, b)

# use a method of myMatrices:
result_own = test.mult()
print(f'result_own: \n{result_own}\n')

# use a method of myMatNew:
totalElem = test.getTotalElem()
print(f'totalElem: {totalElem}\n')
