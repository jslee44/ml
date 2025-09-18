import time
import numpy as np

# List
L = list(range(100000000))
start = time.time()
L_result = [x * 2 for x in L]
print("List:", time.time() - start)

# NumPy
N = np.arange(100000000)
start = time.time()
N_result = N * 2
print(N_result)
print("NumPy:", time.time() - start)