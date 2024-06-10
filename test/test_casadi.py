from casadi import *
import utils
import numpy as np

# TODO: implement error function
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
    curr_ref = traj(t)
    next_ref = traj(t+1)
    curr_ref = np.array(curr_ref)
    next_ref = np.array(next_ref)
    time_interval = utils.time_step
    G_et = np.array([[time_interval * np.cos(curr_error[2]+curr_ref[2]), 0],
                        [time_interval * np.sin(curr_error[2]), 0],
                        [0, time_interval]])
    next_error = curr_error + MX(G_et) @ u_t + (curr_ref-next_ref) + w_t
    return next_error

def cost_fucntion(t, curr_error, next_error, u_t):
    '''
    cost: get the cost at time t
    curr_error: current error [pt_x~, pt_y~, theta_t~]
    next_error: next error [p_t+1_x~, p_t+1_y~, theta_t+1~]
    u_t: control input [v, omega]
    '''
    # TODO: Consider Time Horizon
    TIME_HORIZON = 1
    Q = np.eye(2)
    R = np.eye(2)
    q = 1
    p_err = curr_error[:2]
    theta_err = curr_error[2]
    next_p_err = next_error[:2]
    next_theta_err = next_error[2]
    terminal_cost = next_p_err.T @ Q @ next_p_err + q * np.square(1- np.cos(next_theta_err))
    cost_t = p_err.T @ Q @ p_err + q * np.square(1- np.cos(theta_err)) + u_t.T @ R @ u_t + terminal_cost
    return cost_t


if __name__ == "__main__":
    # Robot info
    traj = utils.lissajous
    U_lb = np.array([0, -1])
    U_ub = np.array([1, 1])
    curr_time = 0
    cur_ref = traj(curr_time)
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    curr_error = np.array(cur_state - cur_ref)
    # Symbols/expressions
    U = MX.sym('U',2,1)
    print("U",U.shape)  
    # uv = U[0]
    # utheata = U[1]
    next_error = error_function(curr_time, MX(curr_error), U, 0)
    print("next error",next_error.shape)
    #y = MX.sym('y')
    #z = MX.sym('z')
    f = cost_fucntion(curr_time, curr_error, next_error, U)

    nlp = {}                 # NLP declaration
    nlp['U']= U # decision vars
    nlp['f'] = f             # objective

    # Create solver instance
    solver = nlpsol("S", "ipopt", nlp)

    # Solve the problem using a guess
    sol = solver(x0=[0.275, 0.0],  # initial guess
                lbU=[0, -1],      # lower bound on optimization variables
                ubU=[1, 1])       # upper bound on optimization variables

    # Solve the problem using a guess
    # sol = solver(x0=[0.275, 0.0], # TODO: initial guess
    #        lbuv=0, # TODO: lower bound on optimization variables
    #        ubuv=1, # TODO: upper bound on optimization variables
    #        lbtheta=-1, # TODO: lower bound on optimization constraints
    #        ubtheta=1, # TODO: upper bound on optimization constraints
    #        )
    x = sol["U"]