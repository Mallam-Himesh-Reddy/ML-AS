lst=[2,7,4,1,3,6]
def sum(lst):
    n=0
    for i in lst:
        for j in lst:
            if i+j==10:
                n+=1
    return n
print(f"{sum(lst)} ")