import numpy as np
def matrix(s,p):
    A = np.zeros((s, s), dtype=int)
    for i in range(s):
        for j in range(s):
            A[i, j] = int(input(f"Enter value for A[{i}][{j}]: "))
    print("\nMatrix A:")
    print(A)

    result = np.linalg.matrix_power(A, p)

    print(f"\nMatrix A^{p}:")
    print(result)

    return result


s = int(input("Enter matrix size: "))
p = int(input("Enter power: "))

matrix(s,p)