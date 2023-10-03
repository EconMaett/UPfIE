import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

gpa3 = woo.dataWoo('gpa3')

# model with full interactions with female dummy (only for spring data):
reg = smf.ols(formula='cumgpa ~ female * (sat + hsperc + tothrs)',
              data=gpa3, subset=(gpa3['spring'] == 1))
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')

# F-Test for H0 (the interaction coefficients of 'female' are zero):
hypotheses = ['female = 0', 'female:sat = 0',
              'female:hsperc = 0', 'female:tothrs = 0']
ftest = results.f_test(hypotheses)
fstat = ftest.statistic[0][0]
fpval = ftest.pvalue

print(f'fstat: {fstat}\n')
print(f'fpval: {fpval}\n')
