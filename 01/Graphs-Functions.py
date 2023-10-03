import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# support of quadratic function
# (creates an array with 100 equispaced elements from -3 to 2):
x1 = np.linspace(-3, 2, num=100)
# function values for all these values:
y1 = x1 ** 2

# plot quadratic function:
plt.plot(x1, y1, linestyle='-', color='black')
plt.savefig('PyGraphs/Graphs-Functions-a.pdf')
plt.close()

# same for normal density:
x2 = np.linspace(-4, 4, num=100)
y2 = stats.norm.pdf(x2)

# plot normal density:
plt.plot(x2, y2, linestyle='-', color='black')
plt.savefig('PyGraphs/Graphs-Functions-b.pdf')
