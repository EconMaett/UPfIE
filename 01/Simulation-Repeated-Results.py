import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

# set the random seed:
np.random.seed(123456)

# set sample size:
n = 100

# initialize ybar to an array of length r=10000 to later store results:
r = 10000
ybar = np.empty(r)

# repeat r times:
for j in range(r):
    # draw a sample and store the sample mean in pos. j=0,1,... of ybar:
    sample = stats.norm.rvs(10, 2, size=n)
    ybar[j] = np.mean(sample)

# the first 20 of 10000 estimates:
print(f'ybar[0:19]: \n{ybar[0:19]}\n')

# simulated mean:
print(f'np.mean(ybar): {np.mean(ybar)}\n')

# simulated variance:
print(f'np.var(ybar, ddof=1): {np.var(ybar, ddof=1)}\n')

# simulated density:
kde = sm.nonparametric.KDEUnivariate(ybar)
kde.fit()

# normal density:
x_range = np.linspace(9, 11)
y = stats.norm.pdf(x_range, 10, np.sqrt(0.04))

# create graph:
plt.plot(kde.support, kde.density, color='black', label='ybar')
plt.plot(x_range, y, linestyle='--', color='black', label='normal distribution')
plt.ylabel('density')
plt.xlabel('ybar')
plt.legend()
plt.savefig('PyGraphs/Simulation-Repeated-Results.pdf')
