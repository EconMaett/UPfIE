import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

# set the random seed:
np.random.seed(1234567)

# set sample size and number of simulations:
n = [5, 10, 100, 1000]
r = 10000

# set true parameters:
beta0 = 1
beta1 = 0.5
sx = 1
ex = 4

for j in n:
    # initialize b1 to store results later:
    b1 = np.empty(r)
    # draw a sample of x, fixed over replications:
    x = stats.norm.rvs(ex, sx, size=j)
    # repeat r times:
    for i in range(r):
        # draw a sample of u (std. normal):
        u = stats.norm.rvs(0, 1, size=j)
        y = beta0 + beta1 * x + u
        df = pd.DataFrame({'y': y, 'x': x})

        # estimate conditional OLS:
        reg = smf.ols(formula='y ~ x', data=df)
        results = reg.fit()
        b1[i] = results.params['x']
    # simulated density:
    kde = sm.nonparametric.KDEUnivariate(b1)
    kde.fit()
    # normal density/ compute mu and se
    X = pd.DataFrame({'const': 1, 'x': x})
    Vbhat = sx * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diagonal(Vbhat))
    x_range = np.linspace(min(b1), max(b1))
    y = stats.norm.pdf(x_range, beta1, se[1])
    # plotting:
    filename = 'PyGraphs/MCSim-olsasy-norm-n' + str(j) + '.pdf'
    # plt.ylim(top=2)
    # plt.xlim(8.5, 11.5)
    plt.plot(kde.support, kde.density, color='black', label='b1')
    plt.plot(x_range, y, linestyle='--', color='black', label='normal distribution')
    plt.ylabel('density')
    plt.xlabel('')
    plt.legend()
    plt.savefig(filename)
    plt.close()
