import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# set the random seed:
np.random.seed(1234567)

# initialize plot:
x_range = np.linspace(0, 50, num=51)
plt.ylim([-18, 18])
plt.xlim([0, 50])

# loop over draws:
for r in range(0, 30):
    # i.i.d. standard normal shock:
    e = stats.norm.rvs(0, 1, size=51)

    # set first entry to 0 (gives y_0 = 0):
    e[0] = 0

    # random walk as cumulative sum of shocks:
    y = np.cumsum(e)

    # add line to graph:
    plt.plot(x_range, y, color='lightgrey', linestyle='-')

plt.axhline(linewidth=2, linestyle='--', color='black')
plt.ylabel('y')
plt.xlabel('time')
plt.savefig('PyGraphs/Simulate-RandomWalk.pdf')
