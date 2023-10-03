import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

rdchem = woo.dataWoo('rdchem')

# OLS regression:
reg = smf.ols(formula='np.log(rd) ~ np.log(sales) + profmarg', data=rdchem)
results = reg.fit()
print(f'results.summary(): \n{results.summary()}\n')

# 95% CI:
CI95 = results.conf_int(0.05)
print(f'CI95: \n{CI95}\n')

# 99% CI:
CI99 = results.conf_int(0.01)
print(f'CI99: \n{CI99}\n')
