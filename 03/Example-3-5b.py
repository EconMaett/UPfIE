import wooldridge as woo
import statsmodels.formula.api as smf

crime1 = woo.dataWoo('crime1')

# model with avgsen:
reg = smf.ols(formula='narr86 ~ pcnv + avgsen + ptime86 + qemp86', data=crime1)
results = reg.fit()
print(f'results.summary(): \n{results.summary()}\n')
