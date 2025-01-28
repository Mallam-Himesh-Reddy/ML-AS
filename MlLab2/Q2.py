lst=[5,3,8,1,0,4]
def find_range(lst):
    min=lst[0]
    max=lst[0]
    for i in lst:
        if min>i:
            min=i
        if max<i:
            max=i
    print(min, max)
    return max-min
print(f" {find_range(lst)}")
