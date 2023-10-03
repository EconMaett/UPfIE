import wooldridge as woo
import numpy as np

ceosal1 = woo.dataWoo('ceosal1')

# extract roe and salary:
roe = ceosal1['roe']
salary = ceosal1['salary']

# sample average:
roe_mean = np.mean(salary)
print(f'roe_mean: {roe_mean}\n')

# sample median:
roe_med = np.median(salary)
print(f'roe_med: {roe_med}\n')

# standard deviation:
roe_s = np.std(salary, ddof=1)
print(f'roe_s: {roe_s}\n')

# correlation with ROE:
roe_corr = np.corrcoef(roe, salary)
print(f'roe_corr: \n{roe_corr}\n')
