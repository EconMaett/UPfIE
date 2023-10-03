import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# support of normal density:
x_range = np.linspace(-4, 4, num=100)

# PDF for all these values:
pdf = stats.norm.pdf(x_range)

# plot:
plt.plot(x_range, pdf, linestyle='-', color='black')
plt.xlabel('x')
plt.ylabel('dx')
plt.savefig('PyGraphs/PDF-example.pdf')
