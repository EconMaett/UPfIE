import wooldridge as woo
import statsmodels.formula.api as smf

gpa1 = woo.dataWoo('gpa1')

# parameter estimates for full and simple model:
reg = smf.ols(formula='colGPA ~ ACT + hsGPA', data=gpa1)
results = reg.fit()
b = results.params
print(f'b: \n{b}\n')

# relation between regressors:
reg_delta = smf.ols(formula='hsGPA ~ ACT', data=gpa1)
results_delta = reg_delta.fit()
delta_tilde = results_delta.params
print(f'delta_tilde: \n{delta_tilde}\n')

# omitted variables formula for b1_tilde:
b1_tilde = b['ACT'] + b['hsGPA'] * delta_tilde['ACT']
print(f'b1_tilde:  \n{b1_tilde}\n')

# actual regression with hsGPA omitted:
reg_om = smf.ols(formula='colGPA ~ ACT', data=gpa1)
results_om = reg_om.fit()
b_om = results_om.params
print(f'b_om: \n{b_om}\n')
