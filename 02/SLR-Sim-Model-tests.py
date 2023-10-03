# XXX im Skript? Fehler in den FOrmeln?
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# set the random seed:
np.random.seed(123456)

# set sample size and number of simulations:
n = 10
r = 10000

# set true parameters:
beta0 = 1
beta1 = 2
su = 0.2
sx = 1
ex = 5

# initialize b0 and b1 to store results later:
b0_uc = np.empty(r)
b1_uc = np.empty(r)
b0_c = np.empty(r)
b1_c = np.empty(r)

# draw a sample of conditional x, fixed over replications:
xc = stats.norm.rvs(ex, sx, size=n)

# repeat r times:
for i in range(r):
    # draw a sample:
    x = stats.norm.rvs(ex, sx, size=n)
    u = stats.norm.rvs(0, su, size=n)
    y = beta0 + beta1 * x + u
    yc = beta0 + beta1 * xc + u
    df = pd.DataFrame({'y': y, 'yc': yc, 'x': x, 'xc': xc})

    # estimate unconditional OLS:
    reg_uc = smf.ols(formula='y ~ x', data=df)
    results_uc = reg_uc.fit()
    b0_uc[i] = results_uc.params[0]
    b1_uc[i] = results_uc.params[1]

    # estimate conditional OLS:
    reg_c = smf.ols(formula='yc ~ xc', data=df)
    results_c = reg_c.fit()
    b0_c[i] = results_c.params[0]
    b1_c[i] = results_c.params[1]
    if (i % 100) == 0:
        print(i)

# comparing theoretical and empirical moments (I):
b0_uc_mean = np.mean(b0_uc)
b1_uc_mean = np.mean(b1_uc)
b0_c_mean = np.mean(b0_c)
b1_c_mean = np.mean(b1_c)

print(f'b0_uc_mean: {b0_uc_mean}\n')
print(f'b0_c_mean: {b0_c_mean}\n')
print(f'b0: {b0}\n')

print(f'b1_uc_mean: {b1_uc_mean}\n')
print(f'b1_c_mean: {b1_c_mean}\n')
print(f'b1: {b1}\n')

# comparing theoretical and empirical moments (II):
b0_uc_var = np.var(b0_uc, ddof=1)
b1_uc_var = np.var(b1_uc, ddof=1)
b0_c_var = np.var(b0_c, ddof=1)
b1_c_var = np.var(b1_c, ddof=1)

x_sq_mean = (sx ** 2) + (ex ** 2)
b0_var = 1 / (n - 1) * (su ** 2) / (sx ** 2) * x_sq_mean
b1_var = 1 / (n - 1) * (su ** 2) / (sx ** 2)

print(f'b0_uc_var: {b0_uc_var}\n')
print(f'b0_c_var: {b0_c_var}\n')
print(f'b0_var: {b0_var}\n')

print(f'b1_uc_var: {b1_uc_var}\n')
print(f'b1_c_var: {b1_c_var}\n')
print(f'b1_var: {b1_var}\n')
