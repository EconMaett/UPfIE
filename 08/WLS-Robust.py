import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

k401ksubs = woo.dataWoo('401ksubs')

# subsetting data:
k401ksubs_sub = k401ksubs[k401ksubs['fsize'] == 1]

# WLS:
wls_weight = list(1 / k401ksubs_sub['inc'])
reg_wls = smf.wls(formula='nettfa ~ inc + I((age-25)**2) + male + e401k',
                  weights=wls_weight, data=k401ksubs_sub)

# non-robust (default) results:
results_wls = reg_wls.fit()
table_default = pd.DataFrame({'b': round(results_wls.params, 4),
                              'se': round(results_wls.bse, 4),
                              't': round(results_wls.tvalues, 4),
                              'pval': round(results_wls.pvalues, 4)})
print(f'table_default: \n{table_default}\n')

# robust results (Refined White SE):
results_white = reg_wls.fit(cov_type='HC3')
table_white = pd.DataFrame({'b': round(results_white.params, 4),
                            'se': round(results_white.bse, 4),
                            't': round(results_white.tvalues, 4),
                            'pval': round(results_white.pvalues, 4)})
print(f'table_white: \n{table_white}\n')
