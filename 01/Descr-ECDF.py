import wooldridge as woo
import numpy as np
import matplotlib.pyplot as plt

ceosal1 = woo.dataWoo('ceosal1')

# extract roe:
roe = ceosal1['roe']

# calculate ECDF:
x = np.sort(roe)
n = x.size
y = np.arange(1, n + 1) / n  # generates cumulative shares of observations

# plot a step function:
plt.step(x, y, linestyle='-', color='black')
plt.xlabel('roe')
plt.savefig('PyGraphs/ecdf.pdf')
