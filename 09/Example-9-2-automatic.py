import wooldridge as woo
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as smo

hprice1 = woo.dataWoo('hprice1')

# original linear regression:
reg = smf.ols(formula='price ~ lotsize + sqrft + bdrms', data=hprice1)
results = reg.fit()

# automated RESET test:
reset_output = smo.reset_ramsey(res=results, degree=3)
fstat_auto = reset_output.statistic[0][0]
fpval_auto = reset_output.pvalue

print(f'fstat_auto: {fstat_auto}\n')
print(f'fpval_auto: {fpval_auto}\n')
