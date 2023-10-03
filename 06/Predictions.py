import wooldridge as woo
import statsmodels.formula.api as smf
import pandas as pd

gpa2 = woo.dataWoo('gpa2')

reg = smf.ols(formula='colgpa ~ sat + hsperc + hsize + I(hsize**2)', data=gpa2)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')

# generate data set containing the regressor values for predictions:
cvalues1 = pd.DataFrame({'sat': [1200], 'hsperc': [30],
                        'hsize': [5]}, index=['newPerson1'])
print(f'cvalues1: \n{cvalues1}\n')

# point estimate of prediction (cvalues1):
colgpa_pred1 = results.predict(cvalues1)
print(f'colgpa_pred1: \n{colgpa_pred1}\n')

# define three sets of regressor variables:
cvalues2 = pd.DataFrame({'sat': [1200, 900, 1400, ],
                        'hsperc': [30, 20, 5], 'hsize': [5, 3, 1]},
                       index=['newPerson1', 'newPerson2', 'newPerson3'])
print(f'cvalues2: \n{cvalues2}\n')

# point estimate of prediction (cvalues2):
colgpa_pred2 = results.predict(cvalues2)
print(f'colgpa_pred2: \n{colgpa_pred2}\n')
