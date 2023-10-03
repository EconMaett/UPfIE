import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

hseinv = woo.dataWoo('hseinv')

# linear regression without time trend:
reg_wot = smf.ols(formula='np.log(invpc) ~ np.log(price)', data=hseinv)
results_wot = reg_wot.fit()

# print regression table:
table_wot = pd.DataFrame({'b': round(results_wot.params, 4),
                          'se': round(results_wot.bse, 4),
                          't': round(results_wot.tvalues, 4),
                          'pval': round(results_wot.pvalues, 4)})
print(f'table_wot: \n{table_wot}\n')

# linear regression with time trend (data set includes a time variable t):
reg_wt = smf.ols(formula='np.log(invpc) ~ np.log(price) + t', data=hseinv)
results_wt = reg_wt.fit()

# print regression table:
table_wt = pd.DataFrame({'b': round(results_wt.params, 4),
                         'se': round(results_wt.bse, 4),
                         't': round(results_wt.tvalues, 4),
                         'pval': round(results_wt.pvalues, 4)})
print(f'table_wt: \n{table_wt}\n')
