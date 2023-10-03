import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

wage1 = woo.dataWoo('wage1')

reg = smf.ols(formula='np.log(wage) ~ educ', data=wage1)
results = reg.fit()
print(f'results.summary(): \n{results.summary()}\n')
