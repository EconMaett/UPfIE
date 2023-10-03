import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# values for x (all between 0 and 10):
x = np.linspace(0, 10, num=11)

# PMF for all these values:
fx = stats.binom.pmf(x, 10, 0.2)

# collect values in DataFrame:
result = pd.DataFrame({'x': x, 'fx': fx})
print(f'result: \n{result}\n')

# plot:
plt.bar(x, fx, color='0.6')
plt.ylabel('x')
plt.ylabel('fx')
plt.savefig('PyGraphs/PMF-example.pdf')
