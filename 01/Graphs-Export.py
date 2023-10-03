import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# support for all normal densities:
x = np.linspace(-4, 4, num=100)

# get different density evaluations:
y1 = stats.norm.pdf(x, 0, 1)
y2 = stats.norm.pdf(x, 0, 3)

# plot (a):
plt.figure(figsize=(4, 6))
plt.plot(x, y1, linestyle='-', color='black')
plt.plot(x, y2, linestyle='--', color='0.3')
plt.savefig('PyGraphs/Graphs-Export-a.pdf')
plt.close()

# plot (b):
plt.figure(figsize=(6, 4))
plt.plot(x, y1, linestyle='-', color='black')
plt.plot(x, y2, linestyle='--', color='0.3')
plt.savefig('PyGraphs/Graphs-Export-b.png')
