import wooldridge as woo
import statsmodels.api as sm
import matplotlib.pyplot as plt

ceosal1 = woo.dataWoo('ceosal1')

# extract roe:
roe = ceosal1['roe']

# estimate kernel density:
kde = sm.nonparametric.KDEUnivariate(roe)
kde.fit()

# subfigure a (kernel density):
plt.plot(kde.support, kde.density, color='black', linewidth=2)
plt.ylabel('density')
plt.xlabel('roe')
plt.savefig('PyGraphs/Density1.pdf')
plt.close()

# subfigure b (kernel density with overlayed histogram):
plt.hist(roe, color='grey', density=True)
plt.plot(kde.support, kde.density, color='black', linewidth=2)
plt.ylabel('density')
plt.xlabel('roe')
plt.savefig('PyGraphs/Density2.pdf')
