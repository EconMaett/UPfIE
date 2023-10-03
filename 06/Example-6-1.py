import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


# define a function for the standardization:
def scale(x):
    x_mean = np.mean(x)
    x_var = np.var(x, ddof=1)
    x_scaled = (x - x_mean) / np.sqrt(x_var)
    return x_scaled


# standardize and estimate:
hprice2 = woo.dataWoo('hprice2')
hprice2['price_sc'] = scale(hprice2['price'])
hprice2['nox_sc'] = scale(hprice2['nox'])
hprice2['crime_sc'] = scale(hprice2['crime'])
hprice2['rooms_sc'] = scale(hprice2['rooms'])
hprice2['dist_sc'] = scale(hprice2['dist'])
hprice2['stratio_sc'] = scale(hprice2['stratio'])

reg = smf.ols(
    formula='price_sc ~ 0 + nox_sc + crime_sc + rooms_sc + dist_sc + stratio_sc',
    data=hprice2)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')
