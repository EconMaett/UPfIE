import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

phillips = woo.dataWoo('phillips')
T = len(phillips)

# define yearly time series beginning in 1948:
date_range = pd.date_range(start='1948', periods=T, freq='Y')
phillips.index = date_range.year

# estimation of expectations-augmented Phillips curve:
yt96 = (phillips['year'] <= 1996)
phillips['inf_diff1'] = phillips['inf'].diff()
reg_ea = smf.ols(formula='inf_diff1 ~ unem', data=phillips, subset=yt96)
results_ea = reg_ea.fit()

phillips['resid_ea'] = results_ea.resid
phillips['resid_ea_lag1'] = phillips['resid_ea'].shift(1)
reg = smf.ols(formula='resid_ea ~ resid_ea_lag1', data=phillips, subset=yt96)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')
