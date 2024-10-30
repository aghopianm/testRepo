def factorial(num):
    if num:
        # call same function by reducing number by 1
        return num  * factorial(num - 1)
    else:
        return 1

res = factorial(5)
print(res)