import numpy as np
# Example 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Mean along axis 1 (rows)
mean = np.mean(arr, axis=0)

# Variance along axis 1 (rows)
variance = np.var(arr, axis=0)

print("Mean:", mean)
print("Variance:", variance)
