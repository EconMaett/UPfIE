import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

gpa3 = woo.dataWoo('gpa3')

# estimate model for males (& spring data):
reg_m = smf.ols(formula='cumgpa ~ sat + hsperc + tothrs',
                data=gpa3,
                subset=(gpa3['spring'] == 1) & (gpa3['female'] == 0))
results_m = reg_m.fit()

# print regression table:
table_m = pd.DataFrame({'b': round(results_m.params, 4),
                        'se': round(results_m.bse, 4),
                        't': round(results_m.tvalues, 4),
                        'pval': round(results_m.pvalues, 4)})
print(f'table_m: \n{table_m}\n')

# estimate model for females (& spring data):
reg_f = smf.ols(formula='cumgpa ~ sat + hsperc + tothrs',
                data=gpa3,
                subset=(gpa3['spring'] == 1) & (gpa3['female'] == 1))
results_f = reg_f.fit()

# print regression table:
table_f = pd.DataFrame({'b': round(results_f.params, 4),
                        'se': round(results_f.bse, 4),
                        't': round(results_f.tvalues, 4),
                        'pval': round(results_f.pvalues, 4)})
print(f'table_f: \n{table_f}\n')
