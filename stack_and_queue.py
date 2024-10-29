lst = [1, 2, 3, 4, 5]
option = input("Stack or queue? (S or Q) ") 

while True:
    choice = int(input("1. Push \n"
               "2. Pop\n"
               "3. View\n"
               "4. Exit "))
    if choice == 1:
        lst.append(int(input("Enter integer ")))
    elif choice == 2 and option == 'Q':
        lst.pop(0)
    elif choice == 2 and option == 'S':
        lst.pop()
    elif choice == 3:
        print(lst)
    elif choice == 4:
        break
    else:
        print("you are not a sigma scientist")