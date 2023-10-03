import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

rdchem = woo.dataWoo('rdchem')

# OLS regression:
reg_ols = smf.ols(formula='rdintens ~ I(sales/1000) + profmarg', data=rdchem)
results_ols = reg_ols.fit()

table_ols = pd.DataFrame({'b': round(results_ols.params, 4),
                          'se': round(results_ols.bse, 4),
                          't': round(results_ols.tvalues, 4),
                          'pval': round(results_ols.pvalues, 4)})
print(f'table_ols: \n{table_ols}\n')

# LAD regression:
reg_lad = smf.quantreg(formula='rdintens ~ I(sales/1000) + profmarg', data=rdchem)
results_lad = reg_lad.fit(q=.5)

table_lad = pd.DataFrame({'b': round(results_lad.params, 4),
                          'se': round(results_lad.bse, 4),
                          't': round(results_lad.tvalues, 4),
                          'pval': round(results_lad.pvalues, 4)})
print(f'table_lad: \n{table_lad}\n')
