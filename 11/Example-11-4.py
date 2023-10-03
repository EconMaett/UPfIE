import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

nyse = woo.dataWoo('nyse')
nyse['ret'] = nyse['return']

# add all lags up to order 3:
nyse['ret_lag1'] = nyse['ret'].shift(1)
nyse['ret_lag2'] = nyse['ret'].shift(2)
nyse['ret_lag3'] = nyse['ret'].shift(3)

# linear regression of model with lags:
reg1 = smf.ols(formula='ret ~ ret_lag1', data=nyse)
reg2 = smf.ols(formula='ret ~ ret_lag1 + ret_lag2', data=nyse)
reg3 = smf.ols(formula='ret ~ ret_lag1 + ret_lag2 + ret_lag3', data=nyse)
results1 = reg1.fit()
results2 = reg2.fit()
results3 = reg3.fit()

# print regression tables:
table1 = pd.DataFrame({'b': round(results1.params, 4),
                       'se': round(results1.bse, 4),
                       't': round(results1.tvalues, 4),
                       'pval': round(results1.pvalues, 4)})
print(f'table1: \n{table1}\n')

table2 = pd.DataFrame({'b': round(results2.params, 4),
                       'se': round(results2.bse, 4),
                       't': round(results2.tvalues, 4),
                       'pval': round(results2.pvalues, 4)})
print(f'table2: \n{table2}\n')

table3 = pd.DataFrame({'b': round(results3.params, 4),
                       'se': round(results3.bse, 4),
                       't': round(results3.tvalues, 4),
                       'pval': round(results3.pvalues, 4)})
print(f'table3: \n{table3}\n')
