import wooldridge as woo
import statsmodels.formula.api as smf
import pandas as pd

gpa2 = woo.dataWoo('gpa2')

reg = smf.ols(formula='colgpa ~ sat + hsperc + hsize + I(hsize**2)', data=gpa2)
results = reg.fit()

# define three sets of regressor variables:
cvalues2 = pd.DataFrame({'sat': [1200, 900, 1400, ],
                        'hsperc': [30, 20, 5], 'hsize': [5, 3, 1]},
                       index=['newPerson1', 'newPerson2', 'newPerson3'])

# point estimates and 95% confidence and prediction intervals:
colgpa_PICI_95 = results.get_prediction(cvalues2).summary_frame(alpha=0.05)
print(f'colgpa_PICI_95: \n{colgpa_PICI_95}\n')

# point estimates and 99% confidence and prediction intervals:
colgpa_PICI_99 = results.get_prediction(cvalues2).summary_frame(alpha=0.01)
print(f'colgpa_PICI_99: \n{colgpa_PICI_99}\n')
