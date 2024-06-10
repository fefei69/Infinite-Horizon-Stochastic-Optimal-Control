from casadi import *
import numpy as np
import utils
# Check collision function using CasADi
traj = utils.lissajous
def check_collision(point):
    """
    Check if the given point collides with any of the circular obstacles.
    
    Args:
    - point: A tuple (x, y) representing the point to check.
    
    Returns:
    - A symbolic expression that evaluates to 1 if the point collides with any obstacle, 0 otherwise.
    """
    obstacles = [(-2, -2, 0.5), (1, 2, 0.5)]
    x, y = point[0], point[1]
    collision = 0
    for (cx, cy, r) in obstacles:
        distance = sqrt((x - cx)**2 + (y - cy)**2)
        collision = collision + if_else(distance <= r, 1, 0)
    return collision

# Error function implementation
def error_function(t, curr_error, u_t, w_t):
    curr_ref = traj(t)
    next_ref = traj(t + 1)
    curr_ref = MX(curr_ref)
    next_ref = MX(next_ref)
    time_interval = utils.time_step
    G_et = vertcat(
        horzcat(time_interval * cos(curr_error[2] + curr_ref[2]), 0),
        horzcat(time_interval * sin(curr_error[2] + curr_ref[2]), 0),
        horzcat(0, time_interval)
    )
    next_error = curr_error + mtimes(G_et, u_t) + (curr_ref - next_ref) + w_t
    return next_error

# Cost function implementation
def cost_function(t, curr_error, next_error, u_t, T):
    traj = utils.lissajous
    q = 1
    Q = MX(np.eye(2))
    R = MX(np.eye(2))
    p_err = curr_error[:2]
    curr_p = p_err + MX(traj(t)[0:2])
    theta_err = curr_error[2]
    next_p_err = next_error[:2]
    next_theta_err = next_error[2]
    # Cost function
    terminal_cost = mtimes(next_p_err.T, mtimes(Q, next_p_err)) + q * (1 - cos(next_theta_err))**2
    cost_t = mtimes(p_err.T, mtimes(Q, p_err)) + q * (1 - cos(theta_err))**2 + mtimes(u_t.T, mtimes(R, u_t)) + terminal_cost
    return cost_t

def CEC(curr_time, cur_state, cur_ref, T):
    # Symbols/expressions
    traj = utils.lissajous
    U = MX.sym('U', 2, T)
    w_t = MX.zeros(3, 1)  # Assuming no noise for now
    curr_error = MX(cur_state - cur_ref)
    f_total = 0
    g = []  # List of constraints

    # Calculate the total cost function over the time horizon T
    for t in range(T):
        next_error = error_function(curr_time + t, curr_error, U[:, t], w_t)
        f_total += cost_function(curr_time + t, curr_error, next_error, U[:, t], T)
        curr_error = next_error
        # Add collision constraints
        curr_p = curr_error[:2] + MX(traj(curr_time + t)[0:2])
        g.append(check_collision(curr_p))

    nlp = {}  # NLP declaration
    nlp['x'] = U.reshape((-1, 1))  # Flatten the decision variable U
    nlp['f'] = f_total  # Total objective
    nlp['g'] = vertcat(*g)  # Constraints

    # Create solver instance
    solver = nlpsol("S", "ipopt", nlp)

    # Set bounds for U iteratively
    x0 = []
    lbx = []
    ubx = []
    for _ in range(T):
        x0.extend([0.2, 0.01])
        lbx.extend([0, -1])  # lower bounds for [v, omega] at each time step
        ubx.extend([1, 1])   # upper bounds for [v, omega] at each time step

    # Set bounds for constraints (0 means no collision allowed)
    lbg = [0] * len(g)
    ubg = [0] * len(g)

    # Flatten the initial guess

    # Solve the problem
    sol = solver(x0=x0,  # initial guess
                 lbx=lbx,  # lower bound on optimization variables
                 ubx=ubx,  # upper bound on optimization variables
                 lbg=lbg,  # lower bound on constraints
                 ubg=ubg)  # upper bound on constraints

    # Extract the solution and reshape it back to (2, T)
    x = sol["x"]
    return np.array(x)  # Convert to a NumPy array


