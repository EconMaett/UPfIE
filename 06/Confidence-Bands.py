import wooldridge as woo
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt

gpa2 = woo.dataWoo('gpa2')

# regress and report coefficients:
reg = smf.ols(formula='colgpa ~ sat', data=gpa2)
results = reg.fit()
print(f'beta_hat: \n{results.params}\n')

# regressor (SAT) values for prediction from 400 to 1600 in steps of 100:
SAT = pd.DataFrame({'sat': np.arange(400, 1600 + 1, 100)})

# predictions and 95% confidence intervals:
colgpa_pred = results.get_prediction(SAT)
colgpa_pred_CI = colgpa_pred.summary_frame(alpha=0.05)[['mean', 'mean_ci_lower', 'mean_ci_upper']]
print(f'colgpa_pred_CI: \n{colgpa_pred_CI}\n')

# plot:
plt.plot(SAT, colgpa_pred_CI['mean'], color='black', linestyle='-', label='')
plt.plot(SAT, colgpa_pred_CI['mean_ci_upper'], color='green', linestyle='--', label='upper CI')
plt.plot(SAT, colgpa_pred_CI['mean_ci_lower'], color='red', linestyle='--', label='lower CI')
plt.ylabel('colgpa')
plt.xlabel('sat')
plt.legend()
plt.savefig('PyGraphs/Confidence-Bands.pdf')

# quadratic model as an alternative:
reg2 = smf.ols(formula='colgpa ~ sat + I(sat**2)', data=gpa2)
results2 = reg2.fit()
colgpa_pred2 = results2.get_prediction(SAT)
colgpa_pred2_CI = colgpa_pred2.summary_frame(alpha=0.05)[['mean', 'mean_ci_lower', 'mean_ci_upper']]

# plot:
plt.plot(SAT, colgpa_pred2_CI['mean'], color='black', linestyle='-', label='')
plt.plot(SAT, colgpa_pred2_CI['mean_ci_upper'], color='green', linestyle='--', label='upper CI')
plt.plot(SAT, colgpa_pred2_CI['mean_ci_lower'], color='red', linestyle='--', label='lower CI')
plt.ylabel('colgpa')
plt.xlabel('sat')
plt.legend()
plt.savefig('PyGraphs/Confidence-Bands2.pdf')
