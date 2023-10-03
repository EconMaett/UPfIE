seq = [1, 2, 3, 4, 5, 6]
for i in range(len(seq)):
    if seq[i] < 4:
        print(seq[i] ** 3)
    else:
        print(seq[i] ** 2)
