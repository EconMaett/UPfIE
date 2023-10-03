import wooldridge as woo
import pandas as pd
import matplotlib.pyplot as plt

barium = woo.dataWoo('barium')
T = len(barium)

# monthly time series starting Feb. 1978:
barium.index = pd.date_range(start='1978-02', periods=T, freq='M')
print(f'barium["chnimp"].head(): \n{barium["chnimp"].head()}\n')

# plot chnimp (default: index on the x-axis):
plt.plot('chnimp', data=barium, color='black', linestyle='-')
plt.ylabel('chnimp')
plt.xlabel('time')
plt.savefig('PyGraphs/Example-Barium.pdf')
