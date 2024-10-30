try:
    num1 = int(input("Enter a number: "))
    num2 = int(input("Enter another number: "))
    
    if num1 > num2:
        if num1 % num2 == 0:
            print(num1, "is a multiple of", num2)
    else:
        if num2 % num1 == 0:
            print(num2, "is a multiple of", num1)

except ZeroDivisionError:
    print("Integer cannot be divisible by zero.")
except ValueError:
    print("Please enter valid integers.")

# Handling input for names and checking for index error
try:
    names = input("List of names: ")
    nameList = names.split()
    print(nameList[1000])  # Accessing an index that may not exist
except IndexError:
    print("The list does not contain enough names.")
except ValueError:
    print("Error processing the names.")

# Handling the randomList conversion to integers
randomList = ['67', 53, '3O', 72, '10']

for i in randomList:
    try:
        print(int(i) * 10)
    except ValueError:
        print(f"Cannot convert '{i}' to an integer.")

# Handling file operations
try:
    fileName = input("Enter File: ")
    with open(fileName) as file:  # Using a context manager to handle file properly
        print(file.read())
except FileNotFoundError:
    print("File not found. Please check the file name and try again.")
except IOError:
    print("An error occurred while reading the file.")