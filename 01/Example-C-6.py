import numpy as np
import scipy.stats as stats

# manually enter raw data from Wooldridge, Table C.3:
SR87 = np.array([10, 1, 6, .45, 1.25, 1.3, 1.06, 3, 8.18, 1.67,
                 .98, 1, .45, 5.03, 8, 9, 18, .28, 7, 3.97])
SR88 = np.array([3, 1, 5, .5, 1.54, 1.5, .8, 2, .67, 1.17, .51,
                 .5, .61, 6.7, 4, 7, 19, .2, 5, 3.83])
Change = SR88 - SR87

# automated calculation of t statistic for H0 (mu=0):
test_auto = stats.ttest_1samp(Change, popmean=0)
t_auto = test_auto.statistic
p_auto = test_auto.pvalue
print(f't_auto: {t_auto}\n')
print(f'p_auto/2: {p_auto / 2}\n')

# manual calculation of t statistic for H0 (mu=0):
avgCh = np.mean(Change)
n = len(Change)
sdCh = np.std(Change, ddof=1)
se = sdCh / np.sqrt(n)
t_manual = avgCh / se
print(f't_manual: {t_manual}\n')

# manual calculation of p value for H0 (mu=0):
p_manual = stats.t.cdf(t_manual, n - 1)
print(f'p_manual: {p_manual}\n')
