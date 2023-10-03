import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

hprice1 = woo.dataWoo('hprice1')

# original OLS:
reg = smf.ols(formula='price ~ lotsize + sqrft + bdrms', data=hprice1)
results = reg.fit()

# regression for RESET test:
hprice1['fitted_sq'] = results.fittedvalues ** 2
hprice1['fitted_cub'] = results.fittedvalues ** 3
reg_reset = smf.ols(formula='price ~ lotsize + sqrft + bdrms +'
                            ' fitted_sq + fitted_cub', data=hprice1)
results_reset = reg_reset.fit()

# print regression table:
table = pd.DataFrame({'b': round(results_reset.params, 4),
                      'se': round(results_reset.bse, 4),
                      't': round(results_reset.tvalues, 4),
                      'pval': round(results_reset.pvalues, 4)})
print(f'table: \n{table}\n')

# RESET test (H0: all coeffs including "fitted" are=0):
hypotheses = ['fitted_sq = 0', 'fitted_cub = 0']
ftest_man = results_reset.f_test(hypotheses)
fstat_man = ftest_man.statistic[0][0]
fpval_man = ftest_man.pvalue

print(f'fstat_man: {fstat_man}\n')
print(f'fpval_man: {fpval_man}\n')
