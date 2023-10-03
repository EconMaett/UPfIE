import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats

# set the random seed:
np.random.seed(123456)

pvals = np.empty(10000)

# repeat r times:
for i in range(10000):
    # i.i.d. N(0,1) innovations:
    n = 51
    e = stats.norm.rvs(0, 1, size=n)
    e[0] = 0
    a = stats.norm.rvs(0, 1, size=n)
    a[0] = 0

    # independent random walks:
    x = np.cumsum(a)
    y = np.cumsum(e)
    sim_data = pd.DataFrame({'y': y, 'x': x})

    # regression:
    reg = smf.ols(formula='y ~ x', data=sim_data)
    results = reg.fit()
    pvals[i] = results.pvalues['x']

# how often is p<=5%:
count_pval_smaller = np.count_nonzero(pvals <= 0.05)  # counts True elements
print(f'count_pval_smaller: {count_pval_smaller}\n')

# how often is p>5%:
count_pval_greater = np.count_nonzero(pvals > 0.05)
print(f'count_pval_greater: {count_pval_greater}\n')
