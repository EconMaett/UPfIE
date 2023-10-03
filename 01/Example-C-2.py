import numpy as np
import scipy.stats as stats

# manually enter raw data from Wooldridge, Table C.3:
SR87 = np.array([10, 1, 6, .45, 1.25, 1.3, 1.06, 3, 8.18, 1.67,
                 .98, 1, .45, 5.03, 8, 9, 18, .28, 7, 3.97])
SR88 = np.array([3, 1, 5, .5, 1.54, 1.5, .8, 2, .67, 1.17, .51,
                 .5, .61, 6.7, 4, 7, 19, .2, 5, 3.83])

# calculate change:
Change = SR88 - SR87

# ingredients to CI formula:
avgCh = np.mean(Change)
print(f'avgCh: {avgCh}\n')

n = len(Change)
sdCh = np.std(Change, ddof=1)
se = sdCh / np.sqrt(n)
print(f'se: {se}\n')

c = stats.t.ppf(0.975, n - 1)
print(f'c: {c}\n')

# confidence interval:
lowerCI = avgCh - c * se
print(f'lowerCI: {lowerCI}\n')

upperCI = avgCh + c * se
print(f'upperCI: {upperCI}\n')
