import wooldridge as woo
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy as pt

hprice1 = woo.dataWoo('hprice1')

# estimate model:
reg = smf.ols(formula='price ~ lotsize + sqrft + bdrms', data=hprice1)
results = reg.fit()
table_results = pd.DataFrame({'b': round(results.params, 4),
                              'se': round(results.bse, 4),
                              't': round(results.tvalues, 4),
                              'pval': round(results.pvalues, 4)})
print(f'table_results: \n{table_results}\n')

# automatic BP test (LM version):
y, X = pt.dmatrices('price ~ lotsize + sqrft + bdrms',
                    data=hprice1, return_type='dataframe')
result_bp_lm = sm.stats.diagnostic.het_breuschpagan(results.resid, X)
bp_lm_statistic = result_bp_lm[0]
bp_lm_pval = result_bp_lm[1]
print(f'bp_lm_statistic: {bp_lm_statistic}\n')
print(f'bp_lm_pval: {bp_lm_pval}\n')

# manual BP test (F version):
hprice1['resid_sq'] = results.resid ** 2
reg_resid = smf.ols(formula='resid_sq ~ lotsize + sqrft + bdrms', data=hprice1)
results_resid = reg_resid.fit()
bp_F_statistic = results_resid.fvalue
bp_F_pval = results_resid.f_pvalue
print(f'bp_F_statistic: {bp_F_statistic}\n')
print(f'bp_F_pval: {bp_F_pval}\n')
