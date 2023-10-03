import wooldridge as woo
import matplotlib.pyplot as plt

ceosal1 = woo.dataWoo('ceosal1')

# extract roe and salary:
roe = ceosal1['roe']
consprod = ceosal1['consprod']

# plotting descriptive statistics:
plt.boxplot(roe, vert=False)
plt.ylabel('roe')
plt.savefig('PyGraphs/Boxplot1.pdf')
plt.close()

# plotting descriptive statistics:
roe_cp0 = roe[consprod == 0]
roe_cp1 = roe[consprod == 1]

plt.boxplot([roe_cp0, roe_cp1])
plt.ylabel('roe')
plt.savefig('PyGraphs/Boxplot2.pdf')
