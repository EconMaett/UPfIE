import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

fertil3 = woo.dataWoo('fertil3')
T = len(fertil3)

# define time series (years only) beginning in 1913:
fertil3.index = pd.date_range(start='1913', periods=T, freq='Y').year

# compute first differences:
fertil3['gfr_diff1'] = fertil3['gfr'].diff()
fertil3['pe_diff1'] = fertil3['pe'].diff()
print(f'fertil3.head(): \n{fertil3.head()}\n')

# linear regression of model with first differences:
reg1 = smf.ols(formula='gfr_diff1 ~ pe_diff1', data=fertil3)
results1 = reg1.fit()

# print regression table:
table1 = pd.DataFrame({'b': round(results1.params, 4),
                       'se': round(results1.bse, 4),
                       't': round(results1.tvalues, 4),
                       'pval': round(results1.pvalues, 4)})
print(f'table1: \n{table1}\n')

# linear regression of model with lagged differences:
fertil3['pe_diff1_lag1'] = fertil3['pe_diff1'].shift(1)
fertil3['pe_diff1_lag2'] = fertil3['pe_diff1'].shift(2)

reg2 = smf.ols(formula='gfr_diff1 ~ pe_diff1 + pe_diff1_lag1 + pe_diff1_lag2',
               data=fertil3)
results2 = reg2.fit()

# print regression table:
table2 = pd.DataFrame({'b': round(results2.params, 4),
                       'se': round(results2.bse, 4),
                       't': round(results2.tvalues, 4),
                       'pval': round(results2.pvalues, 4)})
print(f'table2: \n{table2}\n')
