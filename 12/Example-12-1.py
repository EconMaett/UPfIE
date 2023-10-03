import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

prminwge = woo.dataWoo('prminwge')
T = len(prminwge)
prminwge['time'] = prminwge['year'] - 1949
prminwge.index = pd.date_range(start='1950', periods=T, freq='Y').year

# OLS regression:
reg = smf.ols(formula='np.log(prepop) ~ np.log(mincov) + np.log(prgnp) +'
                      'np.log(usgnp) + time', data=prminwge)

# results with regular SE:
results_regu = reg.fit()

# print regression table:
table_regu = pd.DataFrame({'b': round(results_regu.params, 4),
                           'se': round(results_regu.bse, 4),
                           't': round(results_regu.tvalues, 4),
                           'pval': round(results_regu.pvalues, 4)})
print(f'table_regu: \n{table_regu}\n')

# results with HAC SE:
results_hac = reg.fit(cov_type='HAC', cov_kwds={'maxlags': 2})

# print regression table:
table_hac = pd.DataFrame({'b': round(results_hac.params, 4),
                          'se': round(results_hac.bse, 4),
                          't': round(results_hac.tvalues, 4),
                          'pval': round(results_hac.pvalues, 4)})
print(f'table_hac: \n{table_hac}\n')
