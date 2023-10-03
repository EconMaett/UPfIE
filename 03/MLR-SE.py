import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf

gpa1 = woo.dataWoo('gpa1')

# full estimation results including automatic SE:
reg = smf.ols(formula='colGPA ~ hsGPA + ACT', data=gpa1)
results = reg.fit()

# extract SER (instead of calculation via residuals):
SER = np.sqrt(results.mse_resid)

# regressing hsGPA on ACT for calculation of R2 & VIF:
reg_hsGPA = smf.ols(formula='hsGPA ~ ACT', data=gpa1)
results_hsGPA = reg_hsGPA.fit()
R2_hsGPA = results_hsGPA.rsquared
VIF_hsGPA = 1 / (1 - R2_hsGPA)
print(f'VIF_hsGPA: {VIF_hsGPA}\n')

# manual calculation of SE of hsGPA coefficient:
n = results.nobs
sdx = np.std(gpa1['hsGPA'], ddof=1) * np.sqrt((n - 1) / n)
SE_hsGPA = 1 / np.sqrt(n) * SER / sdx * np.sqrt(VIF_hsGPA)
print(f'SE_hsGPA: {SE_hsGPA}\n')
