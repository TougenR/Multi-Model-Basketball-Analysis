from copy import deepcopy

containments = [
    [0,1],
    [0,4]
]

copy_of_containment = containments.copy()
copy_of_containment[1][1] = 2
print(copy_of_containment)
print("---------------------")
print(containments)

