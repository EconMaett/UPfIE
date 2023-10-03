import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.base.model as smclass

# set the random seed:
np.random.seed(1234567)

x = np.sort(stats.norm.rvs(0, 1, size=100) + 4)
xb = -4 + 1 * x
y_star = xb + stats.norm.rvs(0, 1, size=100)
y = np.copy(y_star)
y[y_star < 0] = 0

x_wc = pd.DataFrame({'const': 1, 'x': x})


# extend statsmodel class:
class Tobit(smclass.GenericLikelihoodModel):
    def nloglikeobs(self, params):
        X = self.exog
        y = self.endog
        p = X.shape[1]
        beta = params[0:p]
        sigma = np.exp(params[p])
        y_hat = np.dot(X, beta)
        y_eq = (y == 0)
        y_g = (y > 0)
        ll = np.empty(len(y))
        ll[y_eq] = np.log(stats.norm.cdf(-y_hat[y_eq] / sigma))
        ll[y_g] = np.log(stats.norm.pdf((y - y_hat)[y_g] / sigma)) - np.log(sigma)
        return -ll


# predictions:
reg_ols = sm.OLS(endog=y, exog=x_wc)
results_ols = reg_ols.fit()
yhat_ols = results_ols.fittedvalues

sigma_start = np.log(sum(results_ols.resid ** 2) / len(results_ols.resid))
params_start = np.concatenate((np.array(results_ols.params), sigma_start), axis=None)
reg_tobit = Tobit(endog=y, exog=x_wc)
results_tobit = reg_tobit.fit(start_params=params_start, disp=0)
yhat_tobit = np.dot(x_wc, np.transpose(results_tobit.params[0:2]))

# plot data and model predictions:
plt.axhline(y=0, linewidth=0.5, linestyle='-', color='grey')
plt.plot(x, y_star, color='black', marker='x',
         linestyle='', label='real data')
plt.plot(x, y, color='black', marker='+',
         linestyle='', label='truncated data')
plt.plot(x, yhat_ols, color='black', marker='',
         linestyle='-', label='OLS fit')
plt.plot(x, yhat_tobit, color='black', marker='',
         linestyle='--', label='Tobit fit')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.savefig('PyGraphs/Tobit-CondMean-Simulation.pdf')
