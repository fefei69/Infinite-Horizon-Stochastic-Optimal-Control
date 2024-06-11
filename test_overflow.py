import numpy as np

test = np.zeros((5,5),dtype=np.float16)
test[0,0] = 1e-10
print(test[0,0])