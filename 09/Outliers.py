import wooldridge as woo
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

rdchem = woo.dataWoo('rdchem')

# OLS regression:
reg = smf.ols(formula='rdintens ~ sales + profmarg', data=rdchem)
results = reg.fit()

# studentized residuals for all observations:
studres = results.get_influence().resid_studentized_external

# display extreme values:
studres_max = np.max(studres)
studres_min = np.min(studres)
print(f'studres_max: {studres_max}\n')
print(f'studres_min: {studres_min}\n')

# histogram (and overlayed density plot):
kde = sm.nonparametric.KDEUnivariate(studres)
kde.fit()

plt.hist(studres, color='grey', density=True)
plt.plot(kde.support, kde.density, color='black', linewidth=2)
plt.ylabel('density')
plt.xlabel('studres')
plt.savefig('PyGraphs/Outliers.pdf')
