import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

phillips = woo.dataWoo('phillips')

# define yearly time series beginning in 1948:
date_range = pd.date_range(start='1948', periods=len(phillips), freq='Y')
phillips.index = date_range.year

# estimate models:
yt96 = (phillips['year'] <= 1996)
reg_1 = smf.ols(formula='unem ~ unem_1', data=phillips, subset=yt96)
results_1 = reg_1.fit()
reg_2 = smf.ols(formula='unem ~ unem_1 + inf_1', data=phillips, subset=yt96)
results_2 = reg_2.fit()

# predictions for 1997-2003 including 95% forecast intervals:
yf97 = (phillips['year'] > 1996)
pred_1 = results_1.get_prediction(phillips[yf97])
pred_1_FI = pred_1.summary_frame(
    alpha=0.05)[['mean', 'obs_ci_lower', 'obs_ci_upper']]
pred_1_FI.index = date_range.year[yf97]
print(f'pred_1_FI: \n{pred_1_FI}\n')

pred_2 = results_2.get_prediction(phillips[yf97])
pred_2_FI = pred_2.summary_frame(
    alpha=0.05)[['mean', 'obs_ci_lower', 'obs_ci_upper']]
pred_2_FI.index = date_range.year[yf97]
print(f'pred_2_FI: \n{pred_2_FI}\n')

# forecast errors:
e1 = phillips[yf97]['unem'] - pred_1_FI['mean']
e2 = phillips[yf97]['unem'] - pred_2_FI['mean']

# RMSE and MAE:
rmse1 = np.sqrt(np.mean(e1 ** 2))
print(f'rmse1: {rmse1}\n')
rmse2 = np.sqrt(np.mean(e2 ** 2))
print(f'rmse2: {rmse2}\n')
mae1 = np.mean(abs(e1))
print(f'mae1: {mae1}\n')
mae2 = np.mean(abs(e2))
print(f'mae2: {mae2}\n')

# graph:
plt.plot(phillips[yf97]['unem'], color='black', marker='', label='unem')
plt.plot(pred_1_FI['mean'], color='black',
         marker='', linestyle='--', label='forecast without inflation')
plt.plot(pred_2_FI['mean'], color='black',
         marker='', linestyle='-.', label='forecast with inflation')
plt.ylabel('unemployment')
plt.xlabel('time')
plt.legend()
plt.savefig('PyGraphs/Example-18-8.pdf')
