import wooldridge as woo
import pandas as pd

lawsch85 = woo.dataWoo('lawsch85')
lsat_pd = lawsch85['LSAT']

# create boolean indicator for missings:
missLSAT = lsat_pd.isna()

# LSAT and indicator for Schools No. 120-129:
preview = pd.DataFrame({'lsat_pd': lsat_pd[119:129],
                        'missLSAT': missLSAT[119:129]})
print(f'preview: \n{preview}\n')

# frequencies of indicator:
freq_missLSAT = pd.crosstab(missLSAT, columns='count')
print(f'freq_missLSAT: \n{freq_missLSAT}\n')

# missings for all variables in data frame (counts):
miss_all = lawsch85.isna()
colsums = miss_all.sum(axis=0)
print(f'colsums: \n{colsums}\n')

# computing amount of complete cases:
complete_cases = (miss_all.sum(axis=1) == 0)
freq_complete_cases = pd.crosstab(complete_cases, columns='count')
print(f'freq_complete_cases: \n{freq_complete_cases}\n')
