# use the predefined class 'list' to create an object:
a = [2, 6, 3, 6]

# access a local variable (to find out what kind of object we are dealing with):
check = type(a).__name__
print(f'check: {check}\n')

# make use of a method (how many 6 are in a?):
count_six = a.count(6)
print(f'count_six: {count_six}\n')

# use another method (sort data in a):
a.sort()
print(f'a: {a}\n')
