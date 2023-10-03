import numpy as np
import scipy.stats as stats
import pandas as pd
import statsmodels.formula.api as smf

# set the random seed:
np.random.seed(1234567)

# set sample size and number of simulations:
n = 1000
r = 10000

# set true parameters (betas):
beta0 = 1
beta1 = 0.5

# initialize b1 arrays to store results later:
b1 = np.empty(r)
b1_me = np.empty(r)

# draw a sample of x, fixed over replications:
xstar = stats.norm.rvs(4, 1, size=n)

# repeat r times:
for i in range(r):
    # draw a sample of u:
    u = stats.norm.rvs(0, 1, size=n)

    # draw a sample of y:
    y = beta0 + beta1 * xstar + u

    # measurement error and mismeasured x:
    e1 = stats.norm.rvs(0, 1, size=n)
    x = xstar + e1
    df = pd.DataFrame({'y': y, 'xstar': xstar, 'x': x})

    # regress y on xstar and store slope estimate at position i:
    reg_star = smf.ols(formula='y ~ xstar', data=df)
    results_star = reg_star.fit()
    b1[i] = results_star.params['xstar']

    # regress y on x and store slope estimate at position i:
    reg_me = smf.ols(formula='y ~ x', data=df)
    results_me = reg_me.fit()
    b1_me[i] = results_me.params['x']

# mean with and without ME:
b1_mean = np.mean(b1)
b1_me_mean = np.mean(b1_me)
print(f'b1_mean: {b1_mean}\n')
print(f'b1_me_mean: {b1_me_mean}\n')

# variance with and without ME:
b1_var = np.var(b1, ddof=1)
b1_me_var = np.var(b1_me, ddof=1)
print(f'b1_var: {b1_var}\n')
print(f'b1_me_var: {b1_me_var}\n')
