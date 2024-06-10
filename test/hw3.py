import numpy as np
# toinv = np.array([[1,-0.9,0,0],[0,1,-0.9,0],[0,0,0.55,-0.45],[0,-0.9,0,1]])
# l = np.array([[-10],[1],[2],[3]])
# v = np.linalg.inv(toinv) @ l
# print(v)
import cvxpy as cp
import numpy as np

# Problem data.
m = 4
n = 1
np.random.seed(1)
# A = np.random.randn(m, n)
w = np.array([[1/4],[1/4],[1/4],[1/4]]).reshape(1,4)

# Construct the problem.
x = cp.Variable(shape=(m, n))
objective = cp.Maximize(w @ x)
constraints = [x[0]<=(-10 + 0.9*x[1]), x[0]<=-8+0.9*(0.5*x[0]+0.5*x[2]),
               x[1]<=(1+0.9*x[2]), x[1]<=(8+0.9*x[1]), 
               x[2]<=(1+0.9*((1/4)*x[0]+(3/4)*x[1])), x[2]<=(2 + 0.9*(0.5*x[2] + 0.5*x[3])),
               x[3]<=(3+0.9*x[1]), x[3]<=(6+0.9*(0.5*x[0]+0.5*x[3]))]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
