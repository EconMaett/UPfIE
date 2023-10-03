import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats

# set the random seed:
np.random.seed(1234567)

# set sample size and number of simulations:
n = 100
r = 10000

# set true parameters:
beta0 = 1
beta1 = 0.5
sx = 1
ex = 4

# initialize b1 to store results later:
b1 = np.empty(r)

# draw a sample of x, fixed over replications:
x = stats.norm.rvs(ex, sx, n)

# repeat r times:
for i in range(r):
    # draw a sample of u (uniform):
    u = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    y = beta0 + beta1 * x + u
    df = pd.DataFrame({'y': y, 'x': x})

    # estimate conditional OLS:
    reg = smf.ols(formula='y ~ x', data=df)
    results = reg.fit()
    b1[i] = results.params['x']
