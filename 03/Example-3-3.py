import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

k401k = woo.dataWoo('401k')

reg = smf.ols(formula='prate ~ mrate + age', data=k401k)
results = reg.fit()
print(f'results.summary(): \n{results.summary()}\n')
