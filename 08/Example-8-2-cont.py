import wooldridge as woo
import statsmodels.formula.api as smf

gpa3 = woo.dataWoo('gpa3')

# definition of model and hypotheses:
reg = smf.ols(formula='cumgpa ~ sat + hsperc + tothrs + female + black + white',
              data=gpa3, subset=(gpa3['spring'] == 1))
hypotheses = ['black = 0', 'white = 0']

# F-Tests using different variance-covariance formulas:
# ususal VCOV:
results_default = reg.fit()
ftest_default = results_default.f_test(hypotheses)
fstat_default = ftest_default.statistic[0][0]
fpval_default = ftest_default.pvalue
print(f'fstat_default: {fstat_default}\n')
print(f'fpval_default: {fpval_default}\n')

# refined White VCOV:
results_hc3 = reg.fit(cov_type='HC3')
ftest_hc3 = results_hc3.f_test(hypotheses)
fstat_hc3 = ftest_hc3.statistic[0][0]
fpval_hc3 = ftest_hc3.pvalue
print(f'fstat_hc3: {fstat_hc3}\n')
print(f'fpval_hc3: {fpval_hc3}\n')

# classical White VCOV:
results_hc0 = reg.fit(cov_type='HC0')
ftest_hc0 = results_hc0.f_test(hypotheses)
fstat_hc0 = ftest_hc0.statistic[0][0]
fpval_hc0 = ftest_hc0.pvalue
print(f'fstat_hc0: {fstat_hc0}\n')
print(f'fpval_hc0: {fpval_hc0}\n')
