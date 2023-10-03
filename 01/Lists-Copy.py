# define a list:
example_list = [1, 5, 41.3, 2.0]

# be careful with changes on variables pointing on example_list:
duplicate_list = example_list
duplicate_list[3] = 10000
print(f'duplicate_list: {duplicate_list}\n')
print(f'example_list: {example_list}\n')

# work on a copy of example_list:
example_list = [1, 5, 41.3, 2.0]
duplicate_list = example_list[:]
duplicate_list[3] = 10000
print(f'duplicate_list: {duplicate_list}\n')
print(f'example_list: {example_list}\n')
