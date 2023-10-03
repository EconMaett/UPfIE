import wooldridge as woo
import numpy as np
import pandas as pd
import scipy.stats as stats

audit = woo.dataWoo('audit')
y = audit['y']

# automated calculation of t statistic for H0 (mu=0):
test_auto = stats.ttest_1samp(y, popmean=0)
t_auto = test_auto.statistic  # access test statistic
p_auto = test_auto.pvalue  # access two-sided p value
print(f't_auto: {t_auto}\n')
print(f'p_auto/2: {p_auto / 2}\n')

# manual calculation of t statistic for H0 (mu=0):
avgy = np.mean(y)
n = len(y)
sdy = np.std(y, ddof=1)
se = sdy / np.sqrt(n)
t_manual = avgy / se
print(f't_manual: {t_manual}\n')

# critical values for t distribution with n-1=240 d.f.:
alpha_one_tailed = np.array([0.1, 0.05, 0.025, 0.01, 0.005, .001])
CV = stats.t.ppf(1 - alpha_one_tailed, 240)
table = pd.DataFrame({'alpha_one_tailed': alpha_one_tailed, 'CV': CV})
print(f'table: \n{table}\n')
