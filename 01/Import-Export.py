import pandas as pd

# import csv with pandas:
df1 = pd.read_csv('data/sales.csv', delimiter=',', header=None,
                  names=['year', 'product1', 'product2', 'product3'])
print(f'df1: \n{df1}\n')

# import txt with pandas:
df2 = pd.read_table('data/sales.txt', delimiter=' ')
print(f'df2: \n{df2}\n')

# add a row to df1:
df3 = df1.append({'year': 2014, 'product1': 10, 'product2': 8, 'product3': 2},
                 ignore_index=True)
print(f'df3: \n{df3}\n')

# export with pandas:
df3.to_csv('data/sales2.csv')
