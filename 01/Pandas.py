import numpy as np
import pandas as pd

# define a pandas DataFrame:
icecream_sales = np.array([30, 40, 35, 130, 120, 60])
weather_coded = np.array([0, 1, 0, 1, 1, 0])
customers = np.array([2000, 2100, 1500, 8000, 7200, 2000])
df = pd.DataFrame({'icecream_sales': icecream_sales,
                   'weather_coded': weather_coded,
                   'customers': customers})

# define and assign an index (six ends of month starting in April, 2010)
# (details on generating indices are given in Chapter 10):
ourIndex = pd.date_range(start='04/2010', freq='M', periods=6)
df.set_index(ourIndex, inplace=True)

# print the DataFrame
print(f'df: \n{df}\n')

# access columns by variable names:
subset1 = df[['icecream_sales', 'customers']]
print(f'subset1: \n{subset1}\n')

# access second to fourth row:
subset2 = df[1:4]  # same as df['2010-05-31':'2010-07-31']
print(f'subset2: \n{subset2}\n')

# access rows and columns by index and variable names:
subset3 = df.loc['2010-05-31', 'customers']  # same as df.iloc[1,2]
print(f'subset3: \n{subset3}\n')

# access rows and columns by index and variable integer positions:
subset4 = df.iloc[1:4, 0:2]
# same as df.loc['2010-05-31':'2010-07-31', ['icecream_sales','weather']]
print(f'subset4: \n{subset4}\n')
