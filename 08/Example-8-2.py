import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

gpa3 = woo.dataWoo('gpa3')

# define regression model:
reg = smf.ols(formula='cumgpa ~ sat + hsperc + tothrs + female + black + white',
              data=gpa3, subset=(gpa3['spring'] == 1))

# estimate default model (only for spring data):
results_default = reg.fit()

table_default = pd.DataFrame({'b': round(results_default.params, 5),
                              'se': round(results_default.bse, 5),
                              't': round(results_default.tvalues, 5),
                              'pval': round(results_default.pvalues, 5)})
print(f'table_default: \n{table_default}\n')

# estimate model with White SE (only for spring data):
results_white = reg.fit(cov_type='HC0')

table_white = pd.DataFrame({'b': round(results_white.params, 5),
                            'se': round(results_white.bse, 5),
                            't': round(results_white.tvalues, 5),
                            'pval': round(results_white.pvalues, 5)})
print(f'table_white: \n{table_white}\n')

# estimate model with refined White SE (only for spring data):
results_refined = reg.fit(cov_type='HC3')

table_refined = pd.DataFrame({'b': round(results_refined.params, 5),
                              'se': round(results_refined.bse, 5),
                              't': round(results_refined.tvalues, 5),
                              'pval': round(results_refined.pvalues, 5)})
print(f'table_refined: \n{table_refined}\n')
