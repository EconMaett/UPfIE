import matplotlib.pyplot as plt

# create data:
x = [1, 3, 4, 7, 8, 9]
y = [0, 3, 6, 9, 7, 8]

# plot and save:
plt.plot(x, y, color='black')
plt.savefig('PyGraphs/Graphs-Basics-a.pdf')
plt.close()
