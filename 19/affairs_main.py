import wooldridge as woo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

affairs = woo.dataWoo('affairs')

# use a pandas.Categorical object to attach labels:
affairs['haskids'] = pd.Categorical.from_codes(affairs['kids'], categories=['no', 'yes'])
counts = affairs['haskids'].value_counts()

# pie chart:
plt.pie(counts, labels=['no', 'yes'])

#

# two variable bar plot:
counts_yes = counts_bykids['yes']
counts_no = counts_bykids['no']
ind = np.arange(len(counts_yes))  # the x locations for the groups
width = 0.4  # width of the bars

# stacked bar plot (c):
plt.bar(['no', 'yes'], counts, color='dimgrey')
plt.bar(ind, counts_no, width, bottom=counts_yes, color='darkgrey', label='No')
plt.ylabel('Counts')
plt.xticks(ind, mlab)
plt.legend()
