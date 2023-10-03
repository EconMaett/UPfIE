import wooldridge as woo
import matplotlib.pyplot as plt

ceosal1 = woo.dataWoo('ceosal1')

# extract roe:
roe = ceosal1['roe']

# subfigure a (histogram with counts):
plt.hist(roe, color='grey')
plt.ylabel('Counts')
plt.xlabel('roe')
plt.savefig('PyGraphs/Histogram1.pdf')
plt.close()

# subfigure b (histogram with density and explicit breaks):
breaks = [0, 5, 10, 20, 30, 60]
plt.hist(roe, color='grey', bins=breaks, density=True)
plt.ylabel('density')
plt.xlabel('roe')
plt.savefig('PyGraphs/Histogram2.pdf')