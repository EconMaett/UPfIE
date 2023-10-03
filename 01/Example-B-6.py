import scipy.stats as stats

# first example using the transformation:
p1_1 = stats.norm.cdf(2 / 3) - stats.norm.cdf(-2 / 3)
print(f'p1_1: {p1_1}\n')

# first example working directly with the distribution of X:
p1_2 = stats.norm.cdf(6, 4, 3) - stats.norm.cdf(2, 4, 3)
print(f'p1_2: {p1_2}\n')

# second example:
p2 = 1 - stats.norm.cdf(2, 4, 3) + stats.norm.cdf(-2, 4, 3)
print(f'p2: {p2}\n')
