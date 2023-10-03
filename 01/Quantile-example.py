import scipy.stats as stats

q_975 = stats.norm.ppf(0.975)
print(f'q_975: {q_975}\n')
