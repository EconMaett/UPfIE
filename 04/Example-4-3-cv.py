import scipy.stats as stats
import numpy as np

# CV for alpha=5% and 1% using the t distribution with 137 d.f.:
alpha = np.array([0.05, 0.01])
cv_t = stats.t.ppf(1 - alpha / 2, 137)
print(f'cv_t: {cv_t}\n')

# CV for alpha=5% and 1% using the normal approximation:
cv_n = stats.norm.ppf(1 - alpha / 2)
print(f'cv_n: {cv_n}\n')
