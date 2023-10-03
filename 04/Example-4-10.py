# Not in use...
#import wooldridge as woo
#import numpy as np
#import statsmodels.formula.api as smf
#import stargazer.stargazer as sg
#
#meap93 = woo.dataWoo('meap93')
#meap93['b_s'] = meap93['benefits'] / meap93['salary']
#
# OLS regression:
#reg1 = smf.ols(formula='np.log(salary) ~ b_s', data=meap93)
#results1 = reg1.fit()
#reg2 = smf.ols(formula='np.log(salary) ~ b_s + np.log(enroll) + np.log(staff)', data=meap93)
#results2 = reg2.fit()
#reg3 = smf.ols(formula='np.log(salary) ~ b_s + np.log(enroll) + np.log(staff) + droprate + gradrate', data=meap93)
#results3 = reg3.fit()
#
## create latex table of results with stargazer
#stargazer = sg.Stargazer([results1, results2, results3])
#print(stargazer.render_latex())
