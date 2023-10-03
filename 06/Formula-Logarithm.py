import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

hprice2 = woo.dataWoo('hprice2')

reg = smf.ols(formula='np.log(price) ~ np.log(nox) + rooms', data=hprice2)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')
