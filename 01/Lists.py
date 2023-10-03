# define a list:
example_list = [1, 5, 41.3, 2.0]
print(f'type(example_list): {type(example_list)}\n')

# access first entry by index:
first_entry = example_list[0]
print(f'first_entry: {first_entry}\n')

# access second to fourth entry by index:
range2to4 = example_list[1:4]
print(f'range2to4: {range2to4}\n')

# replace third entry by new value:
example_list[2] = 3
print(f'example_list: {example_list}\n')

# apply a function:
function_output = min(example_list)
print(f'function_output: {function_output}\n')

# apply a method:
example_list.sort()
print(f'example_list: {example_list}\n')

# delete third element of sorted list:
del example_list[2]
print(f'example_list: {example_list}\n')
