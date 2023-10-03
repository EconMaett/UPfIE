import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

mlb1 = woo.dataWoo('mlb1')

# OLS regression:
reg = smf.ols(
    formula='np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr',
    data=mlb1)
results = reg.fit()

# automated F test:
hypotheses = ['bavg = 0', 'hrunsyr = 2*rbisyr']
ftest = results.f_test(hypotheses)
fstat = ftest.statistic[0][0]
fpval = ftest.pvalue

print(f'fstat: {fstat}\n')
print(f'fpval: {fpval}\n')
