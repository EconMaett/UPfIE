import wooldridge as woo
import numpy as np
import linearmodels.system as iv3

mroz = woo.dataWoo('mroz')

# restrict to non-missing wage observations:
mroz = mroz.dropna(subset=['lwage'])

# 3SLS regressions:
formula = {'eq1': 'hours ~ 1 + educ + age + kidslt6 + nwifeinc +'
                  '[np.log(wage) ~ exper+I(exper**2)]',
           'eq2': 'np.log(wage) ~ 1 + educ + exper + I(exper**2) +'
                  '[hours ~ age + kidslt6 + nwifeinc]'}

reg_3sls = iv3.IV3SLS.from_formula(formula, data=mroz)

results_3sls = reg_3sls.fit(cov_type='unadjusted', debiased=True)
print(f'results_3sls: \n{results_3sls}\n')
