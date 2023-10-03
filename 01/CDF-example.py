import scipy.stats as stats

# binomial CDF:
p1 = stats.binom.cdf(3, 10, 0.2)
print(f'p1: {p1}\n')

# normal CDF:
p2 = stats.norm.cdf(1.96) - stats.norm.cdf(-1.96)
print(f'p2: {p2}\n')
