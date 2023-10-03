import wooldridge as woo
import statsmodels.formula.api as smf
import scipy.stats as stats

mroz = woo.dataWoo('mroz')

# step 1 (use all n observations to estimate a probit model of s_i on z_i):
reg_probit = smf.probit(formula='inlf ~ educ + exper + I(exper**2) +'
                                'nwifeinc + age + kidslt6 + kidsge6',
                        data=mroz)
results_probit = reg_probit.fit(disp=0)
pred_inlf = results_probit.fittedvalues
mroz['inv_mills'] = stats.norm.pdf(pred_inlf) / stats.norm.cdf(pred_inlf)

# step 2 (regress y_i on x_i and inv_mills in sample selection):
reg_heckit = smf.ols(formula='lwage ~ educ + exper + I(exper**2) + inv_mills',
                     subset=(mroz['inlf'] == 1), data=mroz)
results_heckit = reg_heckit.fit()

# print results:
print(f'results_heckit.summary(): \n{results_heckit.summary()}\n')
