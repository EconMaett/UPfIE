import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

mlb1 = woo.dataWoo('mlb1')
n = mlb1.shape[0]

# unrestricted OLS regression:
reg_ur = smf.ols(
    formula='np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr',
    data=mlb1)
fit_ur = reg_ur.fit()
r2_ur = fit_ur.rsquared
print(f'r2_ur: {r2_ur}\n')

# restricted OLS regression:
reg_r = smf.ols(formula='np.log(salary) ~ years + gamesyr', data=mlb1)
fit_r = reg_r.fit()
r2_r = fit_r.rsquared
print(f'r2_r: {r2_r}\n')

# F statistic:
fstat = (r2_ur - r2_r) / (1 - r2_ur) * (n - 6) / 3
print(f'fstat: {fstat}\n')

# CV for alpha=1% using the F distribution with 3 and 347 d.f.:
cv = stats.f.ppf(1 - 0.01, 3, 347)
print(f'cv: {cv}\n')

# p value = 1-cdf of the appropriate F distribution:
fpval = 1 - stats.f.cdf(fstat, 3, 347)
print(f'fpval: {fpval}\n')
