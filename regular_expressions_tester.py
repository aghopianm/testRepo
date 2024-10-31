import re

file = open("The_Raven.txt")

data = file.read()

file.close()

shrieked = re.search("shrieked", data)

bleak = re.search("bleak", data)

pp = len(re.findall(r"\b\w*pp\w*\b", data))

exclamation_mark = re.sub(r"!", "#", data)

starts_and_ends = re.findall(r"\bt\w*(?<!e)\b", data, re.IGNORECASE)

if shrieked:
    print("shrieked is present in the file")
else:
    print("no, shrieked is not present in the file")

if bleak:
    print("bleak is in the file")
else:
    print("no, bleak is not in the file")

if pp:
    print(f"the letters 'pp' have occured {pp} times in the text")
else:
    print("no 'pp' in the file")

if exclamation_mark:
    print("! have now been removed and replaced by #")
else:
    print("This has been unsuccessful")

for i in starts_and_ends:
    print(i)