import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

lawsch85 = woo.dataWoo('lawsch85')

# define cut points for the rank:
cutpts = [0, 10, 25, 40, 60, 100, 175]

# create categorical variable containing ranges for the rank:
lawsch85['rc'] = pd.cut(lawsch85['rank'], bins=cutpts,
                        labels=['(0,10]', '(10,25]', '(25,40]',
                                '(40,60]', '(60,100]', '(100,175]'])

# display frequencies:
freq = pd.crosstab(lawsch85['rc'], columns='count')
print(f'freq: \n{freq}\n')

# run regression:
reg = smf.ols(formula='np.log(salary) ~ C(rc, Treatment("(100,175]")) +'
                      'LSAT + GPA + np.log(libvol) + np.log(cost)',
              data=lawsch85)
results = reg.fit()

# print regression table:
table_reg = pd.DataFrame({'b': round(results.params, 4),
                          'se': round(results.bse, 4),
                          't': round(results.tvalues, 4),
                          'pval': round(results.pvalues, 4)})
print(f'table_reg: \n{table_reg}\n')

# ANOVA table:
table_anova = sm.stats.anova_lm(results, typ=2)
print(f'table_anova: \n{table_anova}\n')
