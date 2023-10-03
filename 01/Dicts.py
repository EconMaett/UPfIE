# define and print a dict:
var1 = ['Florian', 'Daniel']
var2 = [96, 49]
var3 = [True, False]
example_dict = dict(name=var1, points=var2, passed=var3)
print(f'example_dict: \n{example_dict}\n')

# another way to define the dict:
example_dict2 = {'name': var1, 'points': var2, 'passed': var3}
print(f'example_dict2: \n{example_dict2}\n')

# get data type:
print(f'type(example_dict): {type(example_dict)}\n')

# access 'points':
points_all = example_dict['points']
print(f'points_all: {points_all}\n')

# access 'points' of Daniel:
points_daniel = example_dict['points'][1]
print(f'points_daniel: {points_daniel}\n')

# add 4 to 'points' of Daniel and let him pass:
example_dict['points'][1] = example_dict['points'][1] + 4
example_dict['passed'][1] = True
print(f'example_dict: \n{example_dict}\n')

# add a new variable 'grade':
example_dict['grade'] = [1.3, 4.0]

# delete variable 'points':
del example_dict['points']
print(f'example_dict: \n{example_dict}\n')
