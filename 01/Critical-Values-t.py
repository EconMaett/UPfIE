import numpy as np
import pandas as pd
import scipy.stats as stats

# degrees of freedom = n-1:
df = 19

# significance levels:
alpha_one_tailed = np.array([0.1, 0.05, 0.025, 0.01, 0.005, .001])
alpha_two_tailed = alpha_one_tailed * 2

# critical values & table:
CV = stats.t.ppf(1 - alpha_one_tailed, df)
table = pd.DataFrame({'alpha_one_tailed': alpha_one_tailed,
                      'alpha_two_tailed': alpha_two_tailed, 'CV': CV})
print(f'table: \n{table}\n')
