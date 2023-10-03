import wooldridge as woo
import numpy as np
import scipy.stats as stats

audit = woo.dataWoo('audit')
y = audit['y']

# ingredients to CI formula:
avgy = np.mean(y)
n = len(y)
sdy = np.std(y, ddof=1)
se = sdy / np.sqrt(n)
c95 = stats.norm.ppf(0.975)
c99 = stats.norm.ppf(0.995)

# 95% confidence interval:
lowerCI95 = avgy - c95 * se
print(f'lowerCI95: {lowerCI95}\n')

upperCI95 = avgy + c95 * se
print(f'upperCI95: {upperCI95}\n')

# 99% confidence interval:
lowerCI99 = avgy - c99 * se
print(f'lowerCI99: {lowerCI99}\n')

upperCI99 = avgy + c99 * se
print(f'upperCI99: {upperCI99}\n')
