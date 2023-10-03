import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import scipy.stats as stats

# set the random seed:
np.random.seed(123456)

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

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')

# graph:
plt.plot(x, color='black', marker='', linestyle='-', label='x')
plt.plot(y, color='black', marker='', linestyle='--', label='y')
plt.ylabel('x,y')
plt.legend()
plt.savefig('PyGraphs/Simulate-Spurious-Regression-1.pdf')
