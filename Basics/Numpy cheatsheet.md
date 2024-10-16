Here’s a **NumPy Cheatsheet** that highlights the most commonly used functions and operations by data scientists when working with numerical data. This cheatsheet covers array creation, manipulation, mathematical operations, and essential utilities for handling multidimensional arrays.

---

### **1. Importing NumPy**
```python
import numpy as np
```

---

### **2. Creating Arrays**
#### **From Lists or Tuples**
```python
# 1D array
arr1d = np.array([1, 2, 3])

# 2D array
arr2d = np.array([[1, 2], [3, 4]])

# 3D array
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

#### **Zeros, Ones, and Identity Matrix**
```python
# Array of zeros
zeros = np.zeros((2, 3))  # 2x3 matrix of zeros

# Array of ones
ones = np.ones((3, 3))  # 3x3 matrix of ones

# Identity matrix
identity = np.eye(3)  # 3x3 identity matrix
```

#### **Arange and Linspace**
```python
# Array from a range
arange_array = np.arange(0, 10, 2)  # Start from 0 to 9, step of 2

# Evenly spaced numbers
linspace_array = np.linspace(0, 1, 5)  # 5 numbers evenly spaced between 0 and 1
```

#### **Random Arrays**
```python
# Array with random values between 0 and 1
random_array = np.random.rand(3, 3)

# Array with random integers
random_int = np.random.randint(0, 10, (3, 3))  # 3x3 matrix with integers between 0 and 9

# Set seed for reproducibility
np.random.seed(42)
```

---

### **3. Array Properties**
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Shape of the array
print(arr.shape)  # (2, 3)

# Number of dimensions
print(arr.ndim)  # 2

# Data type of elements
print(arr.dtype)  # int64

# Total number of elements
print(arr.size)  # 6
```

---

### **4. Array Indexing & Slicing**
```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Access element at index 2
print(arr[2])  # 3

# Slice array (start:stop:step)
print(arr[1:5])  # [2 3 4 5]

# 2D array indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d[0, 2])  # Element in 1st row, 3rd column

# Slicing rows and columns
print(arr2d[:, 1])  # All rows, 2nd column -> [2 5]
```

---

### **5. Reshaping and Transposing**
#### **Reshape**
```python
arr = np.arange(6)  # [0, 1, 2, 3, 4, 5]

# Reshape into a 2x3 matrix
reshaped = arr.reshape(2, 3)  # [[0, 1, 2], [3, 4, 5]]

# Flatten a multi-dimensional array
flattened = reshaped.flatten()  # [0, 1, 2, 3, 4, 5]
```

#### **Transpose**
```python
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# Transpose the matrix
transposed = arr2d.T  # [[1, 4], [2, 5], [3, 6]]
```

---

### **6. Stacking and Splitting Arrays**
#### **Concatenation**
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenate 1D arrays
concatenated = np.concatenate((arr1, arr2))  # [1 2 3 4 5 6]

# Concatenate 2D arrays along rows (axis=0)
arr2d_1 = np.array([[1, 2], [3, 4]])
arr2d_2 = np.array([[5, 6], [7, 8]])
concatenated_2d = np.concatenate((arr2d_1, arr2d_2), axis=0)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
```

#### **Splitting Arrays**
```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Split into 3 arrays
split = np.split(arr, 3)  # [array([1, 2]), array([3, 4]), array([5, 6])]
```

---

### **7. Mathematical Operations**
#### **Element-wise Operations**
```python
arr = np.array([1, 2, 3, 4])

# Addition, subtraction, multiplication, division
arr_add = arr + 2  # [3, 4, 5, 6]
arr_sub = arr - 1  # [0, 1, 2, 3]
arr_mul = arr * 2  # [2, 4, 6, 8]
arr_div = arr / 2  # [0.5, 1.0, 1.5, 2.0]
```

#### **Aggregate Functions**
```python
arr = np.array([1, 2, 3, 4])

# Sum, mean, min, max
total_sum = np.sum(arr)  # 10
mean_value = np.mean(arr)  # 2.5
min_value = np.min(arr)  # 1
max_value = np.max(arr)  # 4
```

#### **Dot Product**
```python
# Dot product of 1D arrays
arr1 = np.array([1, 2])
arr2 = np.array([3, 4])
dot_product = np.dot(arr1, arr2)  # 1*3 + 2*4 = 11
```

#### **Matrix Multiplication**
```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Matrix multiplication
mat_mul = np.matmul(arr1, arr2)  
# [[19, 22], 
#  [43, 50]]
```

#### **Broadcasting**
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Add a scalar to each element (broadcasting)
arr_broadcast = arr + 10  # [[11, 12, 13], [14, 15, 16]]
```

---

### **8. Statistical Functions**
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Sum across columns (axis=0) or rows (axis=1)
sum_cols = np.sum(arr, axis=0)  # [5, 7, 9]
sum_rows = np.sum(arr, axis=1)  # [6, 15]

# Standard deviation and variance
std_dev = np.std(arr)  # 1.707
variance = np.var(arr)  # 2.916
```

---

### **9. Logical Operations**
```python
arr = np.array([1, 2, 3, 4])

# Boolean condition
cond = arr > 2  # [False, False, True, True]

# Filtering elements
filtered = arr[arr > 2]  # [3, 4]

# Logical AND/OR
logical_and = np.logical_and(arr > 1, arr < 4)  # [False, True, True, False]
```

---

### **10. Handling Missing Data**
```python
arr = np.array([1, 2, np.nan, 4])

# Check for NaN values
nan_mask = np.isnan(arr)  # [False, False, True, False]

# Replace NaN with a specific value
arr_filled = np.nan_to_num(arr, nan=0)  # [1, 2, 0, 4]
```

---

### **11. Copying Arrays**
```python
arr = np.array([1, 2, 3])

# Shallow copy (view)
arr_view = arr.view()

# Deep copy
arr_copy = arr.copy()
```

---

### **12. Linear Algebra**
```python
arr = np.array([[1, 2], [3, 4]])

# Inverse of a matrix
inverse = np.linalg.inv(arr)

# Determinant
determinant = np.linalg.det(arr)  # -2.0

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(arr)
```

---

### **13. Saving and Loading Arrays**
```python
arr = np.array([1, 2, 3])

# Save array to a file
np.save('array.npy', arr)

# Load array from a file
arr_loaded = np.load('array.npy')
```

---

This **NumPy Cheatsheet** provides a comprehensive overview of the most useful operations for manipulating arrays, performing mathematical calculations, reshaping, and handling matrix operations. It is essential for efficient numerical computation, which is frequently used in data science workflows.