import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.stats as stats

# set the random seed:
np.random.seed(1234567)

x = np.sort(stats.norm.rvs(0, 1, size=100) + 4)
y = -4 + 1 * x + stats.norm.rvs(0, 1, size=100)

# complete observations and observed sample:
compl = pd.DataFrame({'x': x, 'y': y})
sample = compl.loc[y > 0]

# predictions OLS:
reg_ols = smf.ols(formula='y ~ x', data=sample)
results_ols = reg_ols.fit()
yhat_ols = results_ols.fittedvalues

# predictions truncated regression:
reg_tr = smf.ols(formula='y ~ x', data=compl)
results_tr = reg_tr.fit()
yhat_tr = results_tr.fittedvalues

# plot data and conditional means:
plt.axhline(y=0, linewidth=0.5, linestyle='-', color='grey')
plt.plot(compl['x'], compl['y'], color='black',
         marker='o', fillstyle='none', linestyle='', label='all data')
plt.plot(sample['x'], sample['y'], color='black',
         marker='o', fillstyle='full', linestyle='', label='sample data')
plt.plot(sample['x'], yhat_ols, color='black',
         marker='', linestyle='--', label='OLS fit')
plt.plot(compl['x'], yhat_tr, color='black',
         marker='', linestyle='-', label='Trunc. Reg. fit')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.savefig('PyGraphs/TruncReg-Simulation.pdf')
