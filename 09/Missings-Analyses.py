import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

lawsch85 = woo.dataWoo('lawsch85')

# missings in numpy:
x_np = np.array(lawsch85['LSAT'])
x_np_bar1 = np.mean(x_np)
x_np_bar2 = np.nanmean(x_np)
print(f'x_np_bar1: {x_np_bar1}\n')
print(f'x_np_bar2: {x_np_bar2}\n')

# missings in pandas:
x_pd = lawsch85['LSAT']
x_pd_bar1 = np.mean(x_pd)
x_pd_bar2 = np.nanmean(x_pd)
print(f'x_pd_bar1: {x_pd_bar1}\n')
print(f'x_pd_bar2: {x_pd_bar2}\n')

# observations and variables:
print(f'lawsch85.shape: {lawsch85.shape}\n')

# regression (missings are taken care of by default):
reg = smf.ols(formula='np.log(salary) ~ LSAT + cost + age', data=lawsch85)
results = reg.fit()
print(f'results.nobs: {results.nobs}\n')
