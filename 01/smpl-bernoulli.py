import scipy.stats as stats

sample = stats.bernoulli.rvs(0.5, size=10)
print(f'sample: {sample}\n')
