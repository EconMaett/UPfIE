import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

intdef = woo.dataWoo('intdef')

# linear regression of static model (Q function avoids conflicts with keywords):
reg = smf.ols(formula='i3 ~ Q("inf") + Q("def")', data=intdef)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.bse, 4),
                      't': round(results.tvalues, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')
