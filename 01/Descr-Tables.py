import wooldridge as woo
import numpy as np
import pandas as pd

affairs = woo.dataWoo('affairs')

# adjust codings to [0-4] (Categoricals require a start from 0):
affairs['ratemarr'] = affairs['ratemarr'] - 1

# use a pandas.Categorical object to attach labels for "haskids":
affairs['haskids'] = pd.Categorical.from_codes(affairs['kids'],
                                               categories=['no', 'yes'])
# ... and "marriage" (for example: 0 = 'very unhappy', 1 = 'unhappy',...):
mlab = ['very unhappy', 'unhappy', 'average', 'happy', 'very happy']
affairs['marriage'] = pd.Categorical.from_codes(affairs['ratemarr'],
                                                categories=mlab)

# frequency table in numpy (alphabetical order of elements):
ft_np = np.unique(affairs['marriage'], return_counts=True)
unique_elem_np = ft_np[0]
counts_np = ft_np[1]
print(f'unique_elem_np: \n{unique_elem_np}\n')
print(f'counts_np: \n{counts_np}\n')

# frequency table in pandas:
ft_pd = affairs['marriage'].value_counts()
print(f'ft_pd: \n{ft_pd}\n')

# frequency table with groupby:
ft_pd2 = affairs['marriage'].groupby(affairs['haskids']).value_counts()
print(f'ft_pd2: \n{ft_pd2}\n')

# contingency table in pandas:
ct_all_abs = pd.crosstab(affairs['marriage'], affairs['haskids'], margins=3)
print(f'ct_all_abs: \n{ct_all_abs}\n')
ct_all_rel = pd.crosstab(affairs['marriage'], affairs['haskids'], normalize='all')
print(f'ct_all_rel: \n{ct_all_rel}\n')

# share within "marriage" (i.e. within a row):
ct_row = pd.crosstab(affairs['marriage'], affairs['haskids'], normalize='index')
print(f'ct_row: \n{ct_row}\n')

# share within "haskids"  (i.e. within a column):
ct_col = pd.crosstab(affairs['marriage'], affairs['haskids'], normalize='columns')
print(f'ct_col: \n{ct_col}\n')
