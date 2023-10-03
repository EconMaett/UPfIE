import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# set the random seed:
np.random.seed(1234567)

x = np.sort(stats.norm.rvs(0, 1, size=100) + 4)
xb = -4 + 1 * x
y_star = xb + stats.norm.rvs(0, 1, size=100)
y = np.copy(y_star)
y[y_star < 0] = 0

# conditional means:
Eystar = xb
Ey = stats.norm.cdf(xb / 1) * xb + 1 * stats.norm.pdf(xb / 1)

# plot data and conditional means:
plt.axhline(y=0, linewidth=0.5,
            linestyle='-', color='grey')
plt.plot(x, y_star, color='black',
         marker='x', linestyle='', label='y*')
plt.plot(x, y, color='black', marker='+',
         linestyle='', label='y')
plt.plot(x, Eystar, color='black', marker='',
         linestyle='-', label='E(y*)')
plt.plot(x, Ey, color='black', marker='',
         linestyle='--', label='E(y)')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.savefig('PyGraphs/Tobit-CondMean.pdf')
