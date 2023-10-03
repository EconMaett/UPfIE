import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# binomial:
# support of binomial PMF:
x_binom = np.linspace(-1, 10, num=1000)

# PMF for all these values:
cdf_binom = stats.binom.cdf(x_binom, 10, 0.2)

# plot:
plt.step(x_binom, cdf_binom, linestyle='-', color='black')
plt.xlabel('x')
plt.ylabel('Fx')
plt.savefig('PyGraphs/CDF-figure-discrete.pdf')
plt.close()

# normal:
# support of normal density:
x_norm = np.linspace(-4, 4, num=1000)

# PDF for all these values:
cdf_norm = stats.norm.cdf(x_norm)

# plot:
plt.plot(x_norm, cdf_norm, linestyle='-', color='black')
plt.xlabel('x')
plt.ylabel('Fx')
plt.savefig('PyGraphs/CDF-figure-cont.pdf')
