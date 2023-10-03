import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

wage1 = woo.dataWoo('wage1')

reg = smf.ols(
    formula='np.log(wage) ~ female + educ + exper + I(exper**2) + tenure + I(tenure**2)',
    data=wage1)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')
