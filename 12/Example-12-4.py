import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

barium = woo.dataWoo('barium')
T = len(barium)

# monthly time series starting Feb. 1978:
barium.index = pd.date_range(start='1978-02', periods=T, freq='M')

reg = smf.ols(formula='np.log(chnimp) ~ np.log(chempi) + np.log(gas) +'
                      'np.log(rtwex) + befile6 + affile6 + afdec6',
              data=barium)
results = reg.fit()

# automatic test:
bg_result = sm.stats.diagnostic.acorr_breusch_godfrey(results, nlags=3)
fstat_auto = bg_result[2]
fpval_auto = bg_result[3]
print(f'fstat_auto: {fstat_auto}\n')
print(f'fpval_auto: {fpval_auto}\n')

# pedestrian test:
barium['resid'] = results.resid
barium['resid_lag1'] = barium['resid'].shift(1)
barium['resid_lag2'] = barium['resid'].shift(2)
barium['resid_lag3'] = barium['resid'].shift(3)

reg_manual = smf.ols(formula='resid ~ resid_lag1 + resid_lag2 + resid_lag3 +'
                             'np.log(chempi) + np.log(gas) + np.log(rtwex) +'
                             'befile6 + affile6 + afdec6', data=barium)
results_manual = reg_manual.fit()

hypotheses = ['resid_lag1 = 0', 'resid_lag2 = 0', 'resid_lag3 = 0']
ftest_manual = results_manual.f_test(hypotheses)
fstat_manual = ftest_manual.statistic[0][0]
fpval_manual = ftest_manual.pvalue
print(f'fstat_manual: {fstat_manual}\n')
print(f'fpval_manual: {fpval_manual}\n')
