import wooldridge as woo
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

hprice1 = woo.dataWoo('hprice1')

# two alternative models:
reg1 = smf.ols(formula='price ~ lotsize + sqrft + bdrms', data=hprice1)
results1 = reg1.fit()

reg2 = smf.ols(formula='price ~ np.log(lotsize) +'
                       'np.log(sqrft) + bdrms', data=hprice1)
results2 = reg2.fit()

# encompassing test of Davidson & MacKinnon:
# comprehensive model:
reg3 = smf.ols(formula='price ~ lotsize + sqrft + bdrms + '
                       'np.log(lotsize) + np.log(sqrft)', data=hprice1)
results3 = reg3.fit()

# model 1 vs. comprehensive model:
anovaResults1 = sm.stats.anova_lm(results1, results3)
print(f'anovaResults1: \n{anovaResults1}\n')

# model 2 vs. comprehensive model:
anovaResults2 = sm.stats.anova_lm(results2, results3)
print(f'anovaResults2: \n{anovaResults2}\n')
