import wooldridge as woo
import statsmodels.formula.api as smf
import scipy.stats as stats

crime1 = woo.dataWoo('crime1')

# 1. estimate restricted model:
reg_r = smf.ols(formula='narr86 ~ pcnv + ptime86 + qemp86', data=crime1)
fit_r = reg_r.fit()
r2_r = fit_r.rsquared
print(f'r2_r: {r2_r}\n')

# 2. regression of residuals from restricted model:
crime1['utilde'] = fit_r.resid
reg_LM = smf.ols(formula='utilde ~ pcnv + ptime86 + qemp86 + avgsen + tottime',
                 data=crime1)
fit_LM = reg_LM.fit()
r2_LM = fit_LM.rsquared
print(f'r2_LM: {r2_LM}\n')

# 3. calculation of LM test statistic:
LM = r2_LM * fit_LM.nobs
print(f'LM: {LM}\n')

# 4. critical value from chi-squared distribution, alpha=10%:
cv = stats.chi2.ppf(1 - 0.10, 2)
print(f'cv: {cv}\n')

# 5. p value (alternative to critical value):
pval = 1 - stats.chi2.cdf(LM, 2)
print(f'pval: {pval}\n')

# 6. compare to F-test:
reg = smf.ols(formula='narr86 ~ pcnv + ptime86 + qemp86 + avgsen + tottime',
              data=crime1)
results = reg.fit()
hypotheses = ['avgsen = 0', 'tottime = 0']
ftest = results.f_test(hypotheses)
fstat = ftest.statistic[0][0]
fpval = ftest.pvalue
print(f'fstat: {fstat}\n')
print(f'fpval: {fpval}\n')
