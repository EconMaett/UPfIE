import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

phillips = woo.dataWoo('phillips')
T = len(phillips)

# define yearly time series beginning in 1948:
date_range = pd.date_range(start='1948', periods=T, freq='Y')
phillips.index = date_range.year

# estimation of static Phillips curve:
yt96 = (phillips['year'] <= 1996)
reg_s = smf.ols(formula='Q("inf") ~ unem', data=phillips, subset=yt96)
results_s = reg_s.fit()

# residuals and AR(1) test:
phillips['resid_s'] = results_s.resid
phillips['resid_s_lag1'] = phillips['resid_s'].shift(1)
reg = smf.ols(formula='resid_s ~ resid_s_lag1', data=phillips, subset=yt96)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')
