import numpy as np
import pandas as pd
import scipy.stats as stats

# nan and inf handling in numpy:
x = np.array([-1, 0, 1, np.nan, np.inf, -np.inf])
logx = np.log(x)
invx = np.array(1 / x)
ncdf = np.array(stats.norm.cdf(x))
isnanx = np.isnan(x)

results = pd.DataFrame({'x': x, 'logx': logx, 'invx': invx,
                        'logx': logx, 'ncdf': ncdf, 'isnanx': isnanx})
print(f'results: \n{results}\n')
