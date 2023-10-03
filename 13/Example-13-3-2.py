import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

kielmc = woo.dataWoo('kielmc')

# difference in difference (DiD):
reg_did = smf.ols(formula='np.log(rprice) ~ nearinc*C(year)', data=kielmc)
results_did = reg_did.fit()

# print regression table:
table_did = pd.DataFrame({'b': round(results_did.params, 4),
                          'se': round(results_did.bse, 4),
                          't': round(results_did.tvalues, 4),
                          'pval': round(results_did.pvalues, 4)})
print(f'table_did: \n{table_did}\n')

# DiD with control variables:
reg_didC = smf.ols(formula='np.log(rprice) ~ nearinc*C(year) + age +'
                           'I(age**2) + np.log(intst) + np.log(land) +'
                           'np.log(area) + rooms + baths',
                   data=kielmc)
results_didC = reg_didC.fit()

# print regression table:
table_didC = pd.DataFrame({'b': round(results_didC.params, 4),
                           'se': round(results_didC.bse, 4),
                           't': round(results_didC.tvalues, 4),
                           'pval': round(results_didC.pvalues, 4)})
print(f'table_didC: \n{table_didC}\n')
