string = input("Enter the string: ")
def c_count(a):
    word=0
    dict = {}
    for i in a:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1
    char=max(dict, key=dict.get)
    return char, dict[char]

char, count = c_count(string)
print(f"{char} with {count} is max.")