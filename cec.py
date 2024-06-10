import casadi
import numpy as np
import utils

class CEC:
    def __init__(self) -> None:
        raise NotImplementedError

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        # TODO: define optimization variables

        # TODO: define optimization constraints and optimization objective

        # TODO: define optimization solver
        nlp = ...
        solver = casadi.nlpsol("S", "ipopt", nlp)
        sol = solver(
            x0=...,  # TODO: initial guess
            lbx=..., # TODO: lower bound on optimization variables
            ubx=..., # TODO: upper bound on optimization variables
            lbg=..., # TODO: lower bound on optimization constraints
            ubg=..., # TODO: upper bound on optimization constraints
        )
        x = sol["x"]  # get the solution

        # TODO: extract the control input from the solution
        u = ...
        return u
    
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
    next_error = curr_error + G_et @ u_t + (curr_ref-next_ref) + w_t
    return next_error

# TODO:test motion model with simple test case
def motion_model(x_t, u_t, w_t):
    '''
    x_t: current state 
    x_t = [pt_x, pt_y, theta]    
    u_t: control input
    u_t = [v, omega]
    w_t: noise (3x1)
    '''
    theta = x_t[2]
    time_interval = utils.time_step
    G_xt = np.array([[time_interval * np.cos(theta), 0],
                     [time_interval * np.sin(theta), 0],
                     [0, time_interval]])
    u_t = np.array(u_t)
    x_t = np.array(x_t)
    next_state = x_t + G_xt @ u_t + w_t
    return next_state

def stage_cost(t, curr_error, u_t):
    '''
    Stage cost: get the cost at time t
    curr_error: current error [pt_x~, pt_y~, theta~]
    '''
    Q = np.eye(3)
    R = np.eye(2)
    q = 1
    p_err = np.array(curr_error[:2])
    theta_err = curr_error[2]
    next_error = error_function(t, curr_error, u_t, 0)
    next_p_err = np.array(next_error[:2])
    next_theta_err = next_error[2]

    cost_t = p_err.T @ Q @ p_err + q * np.square(1- np.cos(theta_err)) + u_t.T @ R @ u_t + \
             next_p_err.T @ Q @ next_p_err + q * np.square(1- np.cos(next_theta_err))
    return cost_t

if __name__ == "__main__":
    print("Testing CEC...")
    traj = utils.lissajous
    sigma = np.array([0.04, 0.04, 0.004]) 
    w_xy = np.random.normal(0, sigma[0], 2)
    w_theta = np.random.normal(0, sigma[2], 1)
    w = np.concatenate((w_xy, w_theta))
    # First Reference state
    curr_time = 0
    cur_ref = traj(curr_time)
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    control = utils.simple_controller(cur_state, cur_ref)
    control = np.array(control).reshape(2,1)
    # print("control after reshape: ", control)
    # control = control[:,0]
    # print("control : ", control)
    print("Testing Motion Model...")
    print("[v,w]", control)
    # Apply Default control input
    next_state_default = utils.car_next_state(utils.time_step, cur_state, control, noise=False)
    next_state = motion_model(cur_state, control, 0)
    print("Default next state: ", next_state_default)
    print("My Next state: ", next_state)
    print("Testing Error Function...")
    curr_error = cur_state - np.array(cur_ref)
    next_error = error_function(0, curr_error, control, 0)
    print("Current error: ", curr_error)
    print("Next error: ", next_error)

    