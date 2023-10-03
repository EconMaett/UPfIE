import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

CPS1985 = pd.read_csv('data/CPS1985.csv')

# run regression:
reg = smf.ols(
    formula='np.log(wage) ~ education + experience + gender + occupation',
    data=CPS1985)
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
