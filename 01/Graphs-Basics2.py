import matplotlib.pyplot as plt

# create data:
x = [1, 3, 4, 7, 8, 9]
y = [0, 3, 6, 9, 7, 8]

# plot and save:
plt.plot(x, y, color='black', linestyle='--')
plt.savefig('PyGraphs/Graphs-Basics-b.pdf')
plt.close()

plt.plot(x, y, color='black', linestyle=':')
plt.savefig('PyGraphs/Graphs-Basics-c.pdf')
plt.close()

plt.plot(x, y, color='black', linestyle='-', linewidth=3)
plt.savefig('PyGraphs/Graphs-Basics-d.pdf')
plt.close()

plt.plot(x, y, color='black', marker='o')
plt.savefig('PyGraphs/Graphs-Basics-e.pdf')
plt.close()

plt.plot(x, y, color='black', marker='v', linestyle='')
plt.savefig('PyGraphs/Graphs-Basics-f.pdf')
