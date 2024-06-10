from casadi import *
import utils
import numpy as np

# Error function implementation
def error_function(t, curr_error, u_t, w_t):
    '''
    Error function: get an estimation of the error at time t+1 
    e_t1 = g(t, e_t, u_t, w_t)
    curr_error: current error [pt_x~, pt_y~, theta~]
    curr_ref: current reference state [rt_x, rt_y, alpha]
    next_ref: next reference state [r_t1_x, r_t1_y, alpha_1]
    u_t = [v, omega]
    w_t: noise (3x1)
    '''
    traj = utils.lissajous
    curr_ref = traj(t)
    next_ref = traj(t+1)
    curr_ref = MX(curr_ref)
    next_ref = MX(next_ref)
    time_interval = utils.time_step
    G_et = vertcat(
        horzcat(time_interval * cos(curr_error[2] + curr_ref[2]), 0),
        horzcat(time_interval * sin(curr_error[2] + curr_ref[2]), 0),
        horzcat(0, time_interval)
    )
    next_error = curr_error + G_et @ u_t + (curr_ref - next_ref) + w_t
    return next_error

def check_collision(point):
    """
    Check if the given point collides with any of the circular obstacles.
    
    Args:
    - point: A tuple (x, y) representing the point to check.
    - obstacles: A list of tuples [(cx1, cy1, r1), (cx2, cy2, r2), ...] representing the circular obstacles.
    
    Returns:
    - True if the point collides with any obstacle, False otherwise.
    """
    obstacles = [(-2, -2, 0.5), (1, 2, 0.5)]
    x, y = point[0], point[1]
    collision = 0
    for (cx, cy, r) in obstacles:
        distance = sqrt((x - cx)**2 + (y - cy)**2)
        collision = if_else(distance <= r, 1, 0)  #if_else(DM cond, DM if_true, DM if_false, bool short_circuit) -> DM
    return collision

# Cost function implementation
def cost_function(t, curr_error, next_error, u_t):
    '''
    cost: get the cost at time t
    curr_error: current error [pt_x~, pt_y~, theta_t~]
    next_error: next error [p_t+1_x~, p_t+1_y~, theta_t+1~]
    u_t: control input [v, omega]
    '''
    traj = utils.lissajous
    q = 1
    Q = MX(np.eye(2))
    R = MX(np.eye(2))
    p_err = curr_error[:2]
    # curr_p = p_err + MX(traj(t)[0:2])
    next_p = next_error[:2] + MX(traj(t+1)[0:2])
    theta_err = curr_error[2]
    next_p_err = next_error[:2]
    next_theta_err = next_error[2]
    # Cost function
    terminal_cost = next_p_err.T @ Q @ next_p_err + q * (1 - cos(next_theta_err))**2
    cost_t = p_err.T @ Q @ p_err + q * (1 - cos(theta_err))**2 + u_t.T @ R @ u_t + terminal_cost
    # Penalize collision
    cost_t += check_collision(next_p)*1e6
    return cost_t

def CEC(curr_time, cur_state, cur_ref, prev_control):
    # Symbols/expressions
    U = MX.sym('U', 2, 1)
    w_t = MX.zeros(3, 1)  # Assuming no noise for now
    curr_error = MX(cur_state - cur_ref)
    next_error = error_function(curr_time, curr_error, U, w_t)
    f = cost_function(curr_time, curr_error, next_error, U)

    nlp = {}  # NLP declaration
    nlp['x'] = U  # decision vars
    nlp['f'] = f  # objective

    # Create solver instance
    solver = nlpsol("S", "ipopt", nlp)

    # Solve the problem using a guess
    sol = solver(x0=[prev_control[0], prev_control[1]],  # initial guess
                 lbx=[0, -1],      # lower bound on optimization variables
                 ubx=[1, 1])       # upper bound on optimization variables

    # Extract the solution
    x = sol["x"]
    return np.array(x) # Default type is casadi.DM

if __name__ == "__main__":
    print("Testing CEC...")
    # Robot info
    traj = utils.lissajous
    curr_time = 0
    cur_ref = traj(curr_time)
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    curr_error = MX(cur_state - cur_ref)
    
    # Symbols/expressions
    U = MX.sym('U', 2, 1)
    w_t = MX.zeros(3, 1)  # Assuming no noise for now
    next_error = error_function(curr_time, curr_error, U, w_t)

    f = cost_function(curr_time, curr_error, next_error, U)

    nlp = {}  # NLP declaration
    nlp['x'] = U  # decision vars
    nlp['f'] = f  # objective

    # Create solver instance
    solver = nlpsol("S", "ipopt", nlp)

    # Solve the problem using a guess
    sol = solver(x0=[0.2, 0.2],  # initial guess
                 lbx=[0, -1],      # lower bound on optimization variables
                 ubx=[1, 1])       # upper bound on optimization variables

    # Extract the solution
    x = sol["x"]
    # print("Optimal solution:", x)

    print("Testing CEC function...")
    control = CEC(curr_time, cur_state, cur_ref)
    print("Control from CEC function:", control)
    print("ontrol from CEC test:", x)
    import pdb; pdb.set_trace()
