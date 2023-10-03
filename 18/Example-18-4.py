import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.api as sm

inven = woo.dataWoo('inven')
inven['lgdp'] = np.log(inven['gdp'])

# automated ADF:
res_ADF_aut = sm.tsa.stattools.adfuller(inven['lgdp'], maxlag=1, autolag=None,
                                        regression='ct', regresults=True)
ADF_stat_aut = res_ADF_aut[0]
ADF_pval_aut = res_ADF_aut[1]
table = pd.DataFrame({'names': res_ADF_aut[3].resols.model.exog_names,
                      'b': np.round(res_ADF_aut[3].resols.params, 4),
                      'se': np.round(res_ADF_aut[3].resols.bse, 4),
                      't': np.round(res_ADF_aut[3].resols.tvalues, 4),
                      'pval': np.round(res_ADF_aut[3].resols.pvalues, 4)})
print(f'table: \n{table}\n')
print(f'ADF_stat_aut: {ADF_stat_aut}\n')
print(f'ADF_pval_aut: {ADF_pval_aut}\n')
