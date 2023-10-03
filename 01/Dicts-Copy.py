# define and print a dict:
var1 = ['Florian', 'Daniel']
var2 = [96, 49]
var3 = [True, False]
example_dict = dict(name=var1, points=var2, passed=var3)
print(f'example_dict: {example_dict}\n')

# if you want to work on a copy:
import copy
copied_dict = copy.deepcopy(example_dict)
copied_dict['points'][1] = copied_dict['points'][1] - 40
print(f'example_dict: \n{example_dict}\n')
print(f'copied_dict: \n{copied_dict}\n')
