# define function:
def mysqrt(x):
    if x >= 0:
        result = x ** 0.5
    else:
        result = 'You fool!'
    return result


# call function and save result:
result1 = mysqrt(4)
print(f'result1: {result1}\n')

result2 = mysqrt(-1.5)
print(f'result2: {result2}\n')
