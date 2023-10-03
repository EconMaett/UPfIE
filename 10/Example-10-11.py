import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

barium = woo.dataWoo('barium')

# linear regression with seasonal effects:
reg = smf.ols(formula='np.log(chnimp) ~ np.log(chempi) + np.log(gas) +'
                      'np.log(rtwex) + befile6 + affile6 + afdec6 +'
                      'feb + mar + apr + may + jun + jul +'
                      'aug + sep + oct + nov + dec',
              data=barium)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')
