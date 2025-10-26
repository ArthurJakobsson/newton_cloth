import numpy as np
import re

array_str = """[ 0.        -0.6070667  0.        -0.6070667  0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        -0.6070667  0.        -0.6070667  0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.       ]"""

# regex that matches integers, decimals with optional trailing/leading digits, and scientific notation
num_re = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

# find all numbers (preserves zeros like "0." and plain "0")
tokens = num_re.findall(array_str)

# convert to floats and keep order
nums = [float(t) for t in tokens]

arr = np.array(nums)
print("Parsed length:", len(arr))

threshold = 0.01
indices = np.where(np.abs(arr) > threshold)[0]

print("Indices with |value| >", threshold, ":", indices.tolist())
print("Values at those indices:", arr[indices].tolist())
