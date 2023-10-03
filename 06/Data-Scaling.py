import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

bwght = woo.dataWoo('bwght')

# regress and report coefficients:
reg = smf.ols(formula='bwght ~ cigs + faminc', data=bwght)
results = reg.fit()

# weight in pounds, manual way:
bwght['bwght_lbs'] = bwght['bwght'] / 16
reg_lbs = smf.ols(formula='bwght_lbs ~ cigs + faminc', data=bwght)
results_lbs = reg_lbs.fit()

# weight in pounds, direct way:
reg_lbs2 = smf.ols(formula='I(bwght/16) ~ cigs + faminc', data=bwght)
results_lbs2 = reg_lbs2.fit()

# packs of cigarettes:
reg_packs = smf.ols(formula='bwght ~ I(cigs/20) + faminc', data=bwght)
results_packs = reg_packs.fit()

# compare results:
table = pd.DataFrame({'b': round(results.params, 4),
                      'b_lbs': round(results_lbs.params, 4),
                      'b_lbs2': round(results_lbs2.params, 4),
                      'b_packs': round(results_packs.params, 4)})
print(f'table: \n{table}\n')
