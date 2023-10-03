import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

##################
## LLN (normal) ##
##################

# set the random seed:
np.random.seed(123456)

# set sample sizes and MC simulations:
n = [10, 50, 100, 1000]
r = 10000

# support of normal density (fixed):
x_range = np.linspace(8.5, 11.5)

for i in n:
    ybar = np.empty(r)
    for j in range(r):
        # sample of size n
        sample = stats.norm.rvs(10, 2, size=i)
        ybar[j] = np.mean(sample)

    # simulated density:
    kde = sm.nonparametric.KDEUnivariate(ybar)
    kde.fit()

    # normal density:
    y = stats.norm.pdf(x_range, 10, 2 / np.sqrt(i))
    # plotting:
    filename = 'PyGraphs/MCSim-lln-n' + str(i) + '.pdf'
    plt.ylim(top=2)
    plt.xlim(8.5, 11.5)
    plt.plot(kde.support, kde.density, color='black', label='ybar')
    plt.plot(x_range, y, linestyle='--', color='black', label='normal distribution')
    plt.ylabel('density')
    plt.xlabel('ybar')
    plt.legend()
    plt.savefig(filename)
    plt.close()

##################
## CLT (chisq) ##
##################

# density:
x_range = np.linspace(0, 3)
y = stats.chi2.pdf(x_range, df=1)
plt.plot(x_range, y, linestyle='-', color='black')
plt.ylabel('density')
plt.savefig('PyGraphs/MCSim-chisqdens.pdf')
plt.close()

# set the random seed:
np.random.seed(123456)

# set sample sizes and MC simulations:
n = [2, 10, 100, 10000]
r = 10000


for i in n:
    ybar = np.empty(r)
    for j in range(r):
        # sample of size n
        sample = stats.chi2.rvs(1, size=i)
        ybar[j] = np.mean(sample)

    # simulated density:
    kde = sm.nonparametric.KDEUnivariate(ybar)
    kde.fit()

    # normal density:
    x_range = np.linspace(min(ybar), max(ybar))
    y = stats.norm.pdf(x_range, 1, np.sqrt(2/i))
    # plotting:
    filename = 'PyGraphs/MCSim-clt-n' + str(i) + '.pdf'
    plt.plot(kde.support, kde.density, color='black', label='ybar')
    plt.plot(x_range, y, linestyle='--', color='black', label='normal distribution')
    plt.ylabel('density')
    plt.xlabel('ybar')
    plt.legend()
    plt.savefig(filename)
    plt.close()
