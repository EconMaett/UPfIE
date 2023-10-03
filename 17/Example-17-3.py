import wooldridge as woo
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

crime1 = woo.dataWoo('crime1')

# estimate linear model:
reg_lin = smf.ols(formula='narr86 ~ pcnv + avgsen + tottime + ptime86 +'
                          'qemp86 + inc86 + black + hispan + born60',
                  data=crime1)
results_lin = reg_lin.fit()

# print regression table:
table_lin = pd.DataFrame({'b': round(results_lin.params, 4),
                          'se': round(results_lin.bse, 4),
                          't': round(results_lin.tvalues, 4),
                          'pval': round(results_lin.pvalues, 4)})
print(f'table_lin: \n{table_lin}\n')

# estimate Poisson model:
reg_poisson = smf.poisson(formula='narr86 ~ pcnv + avgsen + tottime +'
                                  'ptime86 + qemp86 + inc86 + black +'
                                  'hispan + born60',
                          data=crime1)
results_poisson = reg_poisson.fit(disp=0)

# print regression table:
table_poisson = pd.DataFrame({'b': round(results_poisson.params, 4),
                              'se': round(results_poisson.bse, 4),
                              't': round(results_poisson.tvalues, 4),
                              'pval': round(results_poisson.pvalues, 4)})
print(f'table_poisson: \n{table_poisson}\n')

# estimate Quasi-Poisson model:
reg_qpoisson = smf.glm(formula='narr86 ~ pcnv + avgsen + tottime + ptime86 +'
                               'qemp86 + inc86 + black + hispan + born60',
                       family=sm.families.Poisson(),
                       data=crime1)
# the argument scale controls for the dispersion in exponential dispersion models,
# see the module documentation for more details:
results_qpoisson = reg_qpoisson.fit(scale='X2', disp=0)

# print regression table:
table_qpoisson = pd.DataFrame({'b': round(results_qpoisson.params, 4),
                               'se': round(results_qpoisson.bse, 4),
                               't': round(results_qpoisson.tvalues, 4),
                               'pval': round(results_qpoisson.pvalues, 4)})
print(f'table_qpoisson: \n{table_qpoisson}\n')
