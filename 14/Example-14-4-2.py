import wooldridge as woo
import pandas as pd
import linearmodels as plm

wagepan = woo.dataWoo('wagepan')

# estimate different models:
wagepan = wagepan.set_index(['nr', 'year'], drop=False)

reg_ols = plm.PooledOLS.from_formula(
    formula='lwage ~ educ + black + hisp + exper + I(exper**2) +'
            'married + union + C(year)', data=wagepan)
results_ols = reg_ols.fit()

reg_re = plm.RandomEffects.from_formula(
    formula='lwage ~ educ + black + hisp + exper + I(exper**2) +'
            'married + union + C(year)', data=wagepan)
results_re = reg_re.fit()

reg_fe = plm.PanelOLS.from_formula(
    formula='lwage ~ I(exper**2) + married + union +'
            'C(year) + EntityEffects', data=wagepan)
results_fe = reg_fe.fit()

# print results:
theta_hat = results_re.theta.iloc[0, 0]
print(f'theta_hat: {theta_hat}\n')

table_ols = pd.DataFrame({'b': round(results_ols.params, 4),
                          'se': round(results_ols.std_errors, 4),
                          't': round(results_ols.tstats, 4),
                          'pval': round(results_ols.pvalues, 4)})
print(f'table_ols: \n{table_ols}\n')

table_re = pd.DataFrame({'b': round(results_re.params, 4),
                         'se': round(results_re.std_errors, 4),
                         't': round(results_re.tstats, 4),
                         'pval': round(results_re.pvalues, 4)})
print(f'table_re: \n{table_re}\n')

table_fe = pd.DataFrame({'b': round(results_fe.params, 4),
                         'se': round(results_fe.std_errors, 4),
                         't': round(results_fe.tstats, 4),
                         'pval': round(results_fe.pvalues, 4)})
print(f'table_fe: \n{table_fe}\n')
