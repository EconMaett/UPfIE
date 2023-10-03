import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# set the random seed:
np.random.seed(1234567)

# initialize plot:
x_range = np.linspace(1, 50, num=50)
plt.ylim([-1, 5])
plt.xlim([0, 50])

# loop over draws:
for r in range(0, 30):
    # i.i.d. standard normal shock and cumulative sum of shocks:
    e = stats.norm.rvs(0, 1, size=51)
    e[0] = 0
    y = np.cumsum(2 + e)

    # first difference:
    Dy = y[1:51] - y[0:50]

    # add line to graph:
    plt.plot(x_range, Dy, color='lightgrey', linestyle='-')

plt.axhline(y=2, linewidth=2, linestyle='--', color='black')
plt.ylabel('y')
plt.xlabel('time')
plt.savefig('PyGraphs/Simulate-RandomWalkDrift-Diff.pdf')
