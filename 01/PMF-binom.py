import scipy.stats as stats
import math

# pedestrian approach:
c = math.factorial(10) / (math.factorial(2) * math.factorial(10 - 2))
p1 = c * (0.2 ** 2) * (0.8 ** 8)
print(f'p1: {p1}\n')

# scipy function:
p2 = stats.binom.pmf(2, 10, 0.2)
print(f'p2: {p2}\n')
