import numpy as np
import pandas as pd
import pandas_datareader as pdr
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# download data for 'AAPL' (= Apple) and define start and end:
tickers = ['AAPL']
start_date = '2007-12-31'
end_date = '2016-12-31'

# use pandas_datareader for the import:
AAPL_data = pdr.data.DataReader(tickers, 'yahoo', start_date, end_date)

# drop ticker symbol from column name:
AAPL_data.columns = AAPL_data.columns.droplevel(level=1)

# calculate return as the log difference:
AAPL_data['ret'] = np.log(AAPL_data['Adj Close']).diff()

# time series plot of adjusted closing prices:
plt.plot('ret', data=AAPL_data, color='black', linestyle='-')
plt.ylabel('Apple Log Returns')
plt.xlabel('time')
plt.savefig('PyGraphs/Example-EffMkts.pdf')

# linear regression of models with lags:
AAPL_data['ret_lag1'] = AAPL_data['ret'].shift(1)
AAPL_data['ret_lag2'] = AAPL_data['ret'].shift(2)
AAPL_data['ret_lag3'] = AAPL_data['ret'].shift(3)

reg1 = smf.ols(formula='ret ~ ret_lag1', data=AAPL_data)
reg2 = smf.ols(formula='ret ~ ret_lag1 + ret_lag2', data=AAPL_data)
reg3 = smf.ols(formula='ret ~ ret_lag1 + ret_lag2 + ret_lag3', data=AAPL_data)
results1 = reg1.fit()
results2 = reg2.fit()
results3 = reg3.fit()

# print regression tables:
table1 = pd.DataFrame({'b': round(results1.params, 4),
                       'se': round(results1.bse, 4),
                       't': round(results1.tvalues, 4),
                       'pval': round(results1.pvalues, 4)})
print(f'table1: \n{table1}\n')

table2 = pd.DataFrame({'b': round(results2.params, 4),
                       'se': round(results2.bse, 4),
                       't': round(results2.tvalues, 4),
                       'pval': round(results2.pvalues, 4)})
print(f'table2: \n{table2}\n')

table3 = pd.DataFrame({'b': round(results3.params, 4),
                       'se': round(results3.bse, 4),
                       't': round(results3.tvalues, 4),
                       'pval': round(results3.pvalues, 4)})
print(f'table3: \n{table3}\n')
