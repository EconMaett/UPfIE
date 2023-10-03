import numpy as np
import scipy.stats as stats

# sample from a standard normal RV with sample size n=5:
sample1 = stats.norm.rvs(size=5)
print(f'sample1: {sample1}\n')

# a different sample from the same distribution:
sample2 = stats.norm.rvs(size=5)
print(f'sample2: {sample2}\n')

# set the seed of the random number generator and take two samples:
np.random.seed(6254137)
sample3 = stats.norm.rvs(size=5)
print(f'sample3: {sample3}\n')

sample4 = stats.norm.rvs(size=5)
print(f'sample4: {sample4}\n')

# reset the seed to the same value to get the same samples again:
np.random.seed(6254137)
sample5 = stats.norm.rvs(size=5)
print(f'sample5: {sample5}\n')

sample6 = stats.norm.rvs(size=5)
print(f'sample6: {sample6}\n')
