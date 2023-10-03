import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

fertil3 = woo.dataWoo('fertil3')
T = len(fertil3)

# define yearly time series beginning in 1913:
fertil3.index = pd.date_range(start='1913', periods=T, freq='Y').year

# add all lags of 'pe' up to order 2:
fertil3['pe_lag1'] = fertil3['pe'].shift(1)
fertil3['pe_lag2'] = fertil3['pe'].shift(2)

# linear regression of model with lags:
reg = smf.ols(formula='gfr ~ pe + pe_lag1 + pe_lag2 + ww2 + pill', data=fertil3)
results = reg.fit()

# F test (H0: all pe coefficients are=0):
hypotheses1 = ['pe = 0', 'pe_lag1 = 0', 'pe_lag2 = 0']
ftest1 = results.f_test(hypotheses1)
fstat1 = ftest1.statistic[0][0]
fpval1 = ftest1.pvalue

print(f'fstat1: {fstat1}\n')
print(f'fpval1: {fpval1}\n')

# calculating the LRP:
b = results.params
b_pe_tot = b['pe'] + b['pe_lag1'] + b['pe_lag2']
print(f'b_pe_tot: {b_pe_tot}\n')

# F test (H0: LRP=0):
hypotheses2 = ['pe + pe_lag1 + pe_lag2 = 0']
ftest2 = results.f_test(hypotheses2)
fstat2 = ftest2.statistic[0][0]
fpval2 = ftest2.pvalue

print(f'fstat2: {fstat2}\n')
print(f'fpval2: {fpval2}\n')
