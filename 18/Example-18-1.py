import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

hseinv = woo.dataWoo('hseinv')

# add lags and detrend:
hseinv['linvpc_det'] = sm.tsa.tsatools.detrend(hseinv['linvpc'])
hseinv['gprice_lag1'] = hseinv['gprice'].shift(1)
hseinv['linvpc_det_lag1'] = hseinv['linvpc_det'].shift(1)

# Koyck geometric d.l.:
reg_koyck = smf.ols(formula='linvpc_det ~ gprice + linvpc_det_lag1',
                    data=hseinv)
results_koyck = reg_koyck.fit()

# print regression table:
table_koyck = pd.DataFrame({'b': round(results_koyck.params, 4),
                            'se': round(results_koyck.bse, 4),
                            't': round(results_koyck.tvalues, 4),
                            'pval': round(results_koyck.pvalues, 4)})
print(f'table_koyck: \n{table_koyck}\n')

# rational d.l.:
reg_rational = smf.ols(formula='linvpc_det ~ gprice + linvpc_det_lag1 +'
                               'gprice_lag1',
                       data=hseinv)
results_rational = reg_rational.fit()

# print regression table:
table_rational = pd.DataFrame({'b': round(results_rational.params, 4),
                               'se': round(results_rational.bse, 4),
                               't': round(results_rational.tvalues, 4),
                               'pval': round(results_rational.pvalues, 4)})
print(f'table_rational: \n{table_rational}\n')

# LRP:
lrp_koyck = results_koyck.params['gprice'] / (
        1 - results_koyck.params['linvpc_det_lag1'])
print(f'lrp_koyck: {lrp_koyck}\n')

lrp_rational = (results_rational.params['gprice'] +
                results_rational.params['gprice_lag1']) / (
                       1 - results_rational.params['linvpc_det_lag1'])
print(f'lrp_rational: {lrp_rational}\n')
