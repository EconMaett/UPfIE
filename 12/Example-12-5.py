import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy as pt

barium = woo.dataWoo('barium')
T = len(barium)

# monthly time series starting Feb. 1978:
barium.index = pd.date_range(start='1978-02', periods=T, freq='M')

# perform the Cochrane-Orcutt estimation (iterative procedure):
y, X = pt.dmatrices('np.log(chnimp) ~ np.log(chempi) + np.log(gas) +'
                    'np.log(rtwex) + befile6 + affile6 + afdec6',
                    data=barium, return_type='dataframe')
reg = sm.GLSAR(y, X)
CORC_results = reg.iterative_fit(maxiter=100)
table = pd.DataFrame({'b_CORC': CORC_results.params,
                      'se_CORC': CORC_results.bse})
print(f'reg.rho: {reg.rho}\n')
print(f'table: \n{table}\n')
