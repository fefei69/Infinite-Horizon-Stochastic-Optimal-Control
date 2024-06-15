import numpy as np
import utils
from scipy.stats import multivariate_normal
from tqdm import tqdm

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
    # obstacles = [(1, 2, 0.5)]
    x, y = point[0], point[1]
    collision = 0
    for (cx, cy, r) in obstacles:
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        if distance <= r:
            collision += 1
    return collision

def wrap_angle(theta):
    """
    Wraps theta to the range [-pi, pi].
    """
    wrapped_theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return wrapped_theta

def wrap_around_angle_grids(lst):
    """
    Wraps the integer n around the list lst according to the rules provided.
    
    Args:
        lst (list): The list of integers.
        n (int): The integer to wrap around.
        
    Returns:
        int: The wrapped integer.
    """
    # test case
    # lst = np.array([10,11,12,13,14,15,16,17,18,19,20,0,1,2,3,4,5,6,7,8,9])
    rule = np.array([i for i in range(40)])
    invalid_indices = np.where(~np.isin(lst, rule))[0].tolist()
    # If all indices are valid, return the list as it is
    if len(invalid_indices) == 0:
        return lst
    invalid_num = lst[invalid_indices]
    length = len(rule)
    lst[invalid_indices] = rule[(invalid_num % length).tolist()]
    return lst

def map_back_nearest_grid(lst):
    """
    Map the real number back to the nearest grid point.
    a_min and a_max are the lower and upper bound of the grid.
    """ 
    cast_back = np.clip(lst, a_min=0, a_max=20)
    return cast_back

def get_resolution():
    """
    Given Discretized states and controls, compute the resolution of each dimension.
    """
    x_resolution = (x[-1] - x[0]) / (len(x) - 1)
    y_resolution = (y[-1] - y[0]) / (len(y) - 1)
    theta_resolution = (theta[-1] - theta[0]) / (len(theta) - 1)
    t_resolution = (t[-1] - t[0]) / (len(t) - 1)
    v_resolution = (v[-1] - v[0]) / (len(v) - 1)
    w_resolution = (w[-1] - w[0]) / (len(w) - 1)
    return np.array([x_resolution, y_resolution, theta_resolution, t_resolution, v_resolution, w_resolution])


def state_metric_to_index(metric_state: np.ndarray, mode="state_space") -> tuple:
    """
    Convert the metric state to grid indices according to your descretization design.
    Args:
        metric_state (np.ndarray): metric state (x, y, theta, t)
    Returns:
        tuple: grid indices
    """
    res = get_resolution()
    if mode == "state_space":
        # excluded time 
        m = np.array([-3, -3, -np.pi])
        r = res[:3]
    elif mode == "withou_time":
        # This is mutidimensional version, input is |x| x 7 x 3 x |u| 
        m = np.array([-3, -3, -np.pi])[np.newaxis, np.newaxis, :, np.newaxis]
        r = res[:3][np.newaxis, np.newaxis, :, np.newaxis]
    elif mode == "multi-dim_state":
        # This is mutidimensional version, input is |x| x 4 x |u| 
        m = np.array([-3, -3, -np.pi, 0])[np.newaxis, :, np.newaxis]
        r = res[:4][np.newaxis, :, np.newaxis]
    return np.floor((metric_state-m)/r).astype(int)

def state_index_to_metric(state_index: tuple) -> np.ndarray:
    """
    Convert the grid indices to metric state according to your descretization design.
    Args:
        state_index (tuple): grid indices
    Returns:
        np.ndarray: metric state
    """
    res = get_resolution()
    m = np.array([-3, -3, -np.pi])
    r = res[:3]
    return state_index * r + m

def control_metric_to_index(control_metric: np.ndarray) -> tuple:
    """
    Args:
        control_metric: [2, N] array of controls in metric space
    Returns:
        [N, ] array of indices in the control space
    """
    m = np.array([0, -1])
    res = get_resolution()
    r = res[4:]
    return np.floor((control_metric-m)/r).astype(int)


def control_index_to_metric(control_in_grids):
    m = np.array([0, -1])
    res = get_resolution()
    r = res[4:]
    return (control_in_grids)*r+m


def get_neighbors(state: np.ndarray) -> tuple:
    # state: (|x|,4,|u|)
    # Resolution of the grid
    res = get_resolution()
    r = res[:3]

    # Split next_error into components
    x, y, theta, t = state[:, 0, :], state[:, 1, :], state[:, 2, :], state[:, 3, :]
    dim_states = x.shape[0]
    dim_controls = x.shape[1]
    # Initialize neighbors and neighbors_index arrays
    neighbors = np.zeros((dim_states, 7, 3, dim_controls))
    neighbors_index = np.zeros((dim_states, 7, 3, dim_controls), dtype=int)

    # Calculate neighbors
    neighbors[:, 0, :, :] = state[:, :3, :]  # state
    neighbors[:, 1, :, :] = state[:, :3, :] + np.array([r[0], 0, 0])[:, np.newaxis]  # +x
    neighbors[:, 2, :, :] = state[:, :3, :] + np.array([-r[0], 0, 0])[:, np.newaxis]  # -x
    neighbors[:, 3, :, :] = state[:, :3, :] + np.array([0, r[1], 0])[:, np.newaxis]  # +y
    neighbors[:, 4, :, :] = state[:, :3, :] + np.array([0, -r[1], 0])[:, np.newaxis]  # -y
    neighbors[:, 5, :, :] = state[:, :3, :] + np.array([0, 0, r[2]])[:, np.newaxis]  # +theta
    neighbors[:, 6, :, :] = state[:, :3, :] + np.array([0, 0, -r[2]])[:, np.newaxis]  # -theta

    neighbors_index = state_metric_to_index(neighbors, mode="withou_time")
    return neighbors, neighbors_index

def normalize_prob(probabilities: np.ndarray) -> np.ndarray:
    # Define a threshold
    threshold = 1e-10
    # Set probabilities smaller than the threshold to zero
    probabilities[probabilities < threshold] = 0
    # Sum of remaining probabilities
    total_prob = np.sum(probabilities,axis=1, keepdims=True)
    return probabilities/total_prob

def add_noise(mean, samples):
    # mean: |x| x 4 x |u|
    # samples: |x| x 7 x 3 x |u|
    dim_states = mean.shape[0]
    dim_controls = mean.shape[2]
    res = get_resolution()
    r = np.square(res[:3])
    # rescaled sigma to grids unit
    sigma = np.array([[0.04, 0, 0],
                      [0, 0.04, 0],
                      [0, 0, 0.004]])
    res = get_resolution()
    r = res[:3]
    # import pdb; pdb.set_trace()
    mean, t = mean[:,:3,:], mean[:,3,:]
    samples = np.array(samples)
    # Reshape mean and samples for vectorized operations
    mean_pos_reshaped = mean.transpose(0, 2, 1).reshape(-1, 3)  # Shape: (|x|*|u|, 3)
    samples_reshaped = samples.transpose(0, 3, 1, 2).reshape(-1, 7, 3)  # Shape: (|x|*|u|, 7, 3)

    # Compute the differences between samples and means
    diffs = samples_reshaped - mean_pos_reshaped[:, np.newaxis, :]  # Shape: (|x|*|u|, 7, 3)

    # Compute the exponents in a vectorized manner
    inv_sigma = np.linalg.inv(sigma)  # Shape: (3, 3)
    exponents = -0.5 * np.sum(np.matmul(diffs, inv_sigma) * diffs, axis=-1)  # Shape: (|x|*|u|, 7)

    # Compute the normalization factor
    normalization_factor = np.sqrt((2 * np.pi) ** 3 * np.linalg.det(sigma))

    # Compute probabilities
    probabilities = np.exp(exponents) / normalization_factor  # Shape: (|x|*|u|, 7)
    # import pdb; pdb.set_trace()
    # Normalize probabilities
    probabilities = normalize_prob(probabilities).reshape(dim_states, dim_controls, 7, 1)  # Shape: (|x|, |u|, 7, 1)
    # import pdb; pdb.set_trace()
    # Concatenate probabilities to samples
    samples_reshaped = samples.transpose(0, 3, 1, 2).reshape(dim_states, dim_controls, 7, 3)  # Shape: (|x|, |u|, 7, 3)
    full_samples = np.concatenate((samples_reshaped, probabilities), axis=3)  # Shape: (|x|, |u|, 7, 4)

    # Transpose to desired output shape
    return full_samples.transpose(0, 1, 2, 3)  # Shape: (|x|, |u|, 7, 4)
    

def error_function(t, curr_error, u_t, noise = True):
    '''
    Error function: get an estimation of the error at time t+1 
    e_t1 = g(t, e_t, u_t, w_t)
    curr_error: current error [pt_x~, pt_y~, theta~]
    curr_ref: current reference state [rt_x, rt_y, alpha]
    next_ref: next reference state [r_t1_x, r_t1_y, alpha_1]
    u_t = [v, omega]
    w_t: noise (3x1)
    '''
    assert noise==True, "have not implement the version of 0 noise"
    traj = utils.lissajous
    curr_ref = np.array(traj(t))
    next_ref = np.array(traj(t+1))
    time_interval = utils.time_step

    G_et = np.stack([
                    np.stack([time_interval * np.cos(curr_error[:, 2] + curr_ref[2]), np.zeros_like(curr_error[:, 2])], axis=-1),
                    np.stack([time_interval * np.sin(curr_error[:, 2]), np.zeros_like(curr_error[:, 2])], axis=-1),
                    np.stack([np.zeros_like(curr_error[:, 2]), time_interval * np.ones_like(curr_error[:, 2])], axis=-1)
                    ], axis=1)
    # Compute the next error at time t+1
    ref_error = curr_ref - next_ref
    next_error = curr_error[:, :, np.newaxis] + np.einsum('ijk,kl->ijl', G_et, u_t.T) + ref_error[np.newaxis, :, np.newaxis]
    # Wrap the angle
    next_error[:,2,:] = wrap_angle(next_error[:,2,:])
    new_time = np.full((next_error.shape[0], 1, next_error.shape[2]), t+1)
    next_error = np.concatenate((next_error, new_time), axis=1) 
    # Get the 7 neighbors of the next error
    neighbors, neighbors_index = get_neighbors(next_error)
    # Consider the noise and get the probabilities of transitioning to each neighbor
    next_error = state_metric_to_index(next_error, mode="multi-dim_state")
    probabilities_matrix = add_noise(next_error, neighbors_index)
    return probabilities_matrix 
    
    
def compute_transition_probability(control_space, state_space):
    # Initialize the transition probability matrix P (t, x, y, theta, v, w, 7, 4)
    P = np.zeros((100, 21, 21, 40, 6, 11, 7, 4))
    for t in tqdm(range(100)):
        P[t,:,:,:,:,:,:,:] = error_function(t, state_space[:,:3], control_space, noise=True).reshape((21, 21, 40, 6, 11, 7, 4))
    # print("Testing prob",P[1,2,3,4,5,6,:,:])
    # import pdb; pdb.set_trace()
    return P
    # np.save("transition_matrix_100iters_larger_space.npy",P)
# Transition matrix
# P_stored = np.load('transition_matrix_10iters.npy')
def compute_stage_cost(control_space, state_space):
    """
    Compute the stage cost L(t, x, y, theta, t, v, w) for all states and controls.
    """
    # traj = utils.lissajous
    q = 1
    Q = np.eye(2)
    R = np.eye(2)
    # w_t = np.zeros((3, 1))
    L = np.zeros((100, 21, 21, 40, 6, 11))
    pos_err = state_space[:,:2] # (1210,2)
    theta_err = state_space[:,2] # (1210,)
    u_t = control_space # (66,2)
    # import pdb; pdb.set_trace()
    # g = []
    print("Computing stage cost...")
    for t in tqdm(range(100)):
        # Expand dimensions of pos_err and u_t for broadcasting
        pos_err_expanded = pos_err[:, np.newaxis, :]  # Shape (1210, 1, 2)
        u_t_expanded = u_t[:, np.newaxis, :]  # Shape (1, 66, 2)
        # (1210, 1, 2) @ (2, 2) @ (1210, 1, 2)^T -> (1210, 66)
        # Compute position error cost
        pos_err_cost = np.einsum('nij,jk,nkj->ni', pos_err_expanded, Q, pos_err_expanded)  # Shape (1210, 1)

        # Compute angle error cost
        # Shape (1210, 1), broadcastable to (1210, 66)
        theta_err_cost = q * (1 - np.cos(theta_err[:, np.newaxis]))**2  # Shape (1210, 1), broadcastable to (1210, 66)

        # Compute control input cost
        #(1, 66, 2) @ (2, 2) @ (1, 66, 2)^T -> (66)
        u_t_cost = np.einsum('nij,jk,nkj->ni', u_t_expanded, R, u_t_expanded).reshape(1,-1)  # Shape (1, 66)
        # Combine all costs
        total_cost = pos_err_cost + theta_err_cost + u_t_cost  # Shape (1210, 66)
        L[t,:,:,:,:,:] = total_cost.reshape((21, 21, 40, 6, 11))
        # import pdb; pdb.set_trace()
        # Penalize collision
        # cost_t += check_collision(curr_p)*1000
        # g.append(check_collision(next_p))
    # import pdb; pdb.set_trace()
    return L

def policy_iteration(control_space, state_space, num_iter=50):
    # t Value functions V(x)
    V = np.zeros((100, 21, 21, 40)) 
    P = np.load('transition_matrix_100iters_larger_space_larger_prob.npy') # (100, 11, 11, 10, 6, 11, 7, 4)
    L = compute_stage_cost(control_space, state_space) # (100, 11, 11, 10, 6, 11)
    pi = np.zeros((100, 21, 21, 40, 2),dtype='int')
    # Discount factor
    gamma = 0.95
    # t = 0
    # GPI
    for t in tqdm(range(30)):
        for itr in range(50): #TODO: change this to while True, since sometimes V does not converge
            print("PI Iteration: ", itr)
            # Policy Evaluation
            for _ in range(num_iter):
                OLD_V = V.copy()
                for ex in range(21):
                    for ey in range(21):
                        for theta in range(40):
                            v_control = pi[t,ex,ey,theta,0]
                            w_control = pi[t,ex,ey,theta,1]
                            # print("[V,w]",v_control, w_control)
                            x_indices = map_back_nearest_grid(P[t,ex,ey,theta,v_control,w_control,:,0].astype('int'))
                            y_indices = map_back_nearest_grid(P[t,ex,ey,theta,v_control,w_control,:,1].astype('int'))
                            theta_indices = wrap_around_angle_grids(P[t,ex,ey,theta,v_control,w_control,:,2].astype('int'))
                            try:
                                V[t,ex,ey,theta] = L[t,ex,ey,theta,v_control,w_control] + gamma * np.sum(P[t,ex,ey,theta,v_control,w_control,:,3] * V[t, x_indices, y_indices, theta_indices])
                                # import pdb; pdb.set_trace()
                            except IndexError:
                                print("Some Indices are out of bounds")
                                import pdb; pdb.set_trace()
                # print("Diference in V: ", np.linalg.norm(V - OLD_V))
                if np.allclose(V, OLD_V):
                    print("Converged")
                    break
            # import pdb; pdb.set_trace()
            # Policy Improvement
            for ex in range(21):
                for ey in range(21):
                    for theta in range(40):
                        x_indices = map_back_nearest_grid(P[t,ex,ey,theta,:,:,:,0].astype('int'))
                        y_indices = map_back_nearest_grid(P[t,ex,ey,theta,:,:,:,1].astype('int'))
                        theta_indices = wrap_around_angle_grids(P[t,ex,ey,theta,:,:,:,2].astype('int'))
                        Q = L[t,ex,ey,theta,:,:] + gamma * np.sum(P[t,ex,ey,theta,:,:,:,3] * V[t, x_indices, y_indices, theta_indices],axis=2)
                        # import pdb; pdb.set_trace()
                        pi[t, ex, ey, theta, :] = np.unravel_index(np.argmin(Q), Q.shape) 
                        # print("Q",Q)
                        # import pdb; pdb.set_trace()
    return pi
# Discount factor
gamma = 0.95

# Stage cost
# L = np.zeros((10,11,11,10)) 

x = np.linspace(-3, 3, 21)
y = np.linspace(-3, 3, 21)
theta = np.linspace(-np.pi, np.pi, 40, endpoint=False)  #[-pi, pi]
t = np.linspace(0, 99, 100)

# Create Contol Space
v = np.linspace(0, 1, 6)
w = np.linspace(-1, 1, 11)

# Create the 3D grid using meshgrid
# xx, yy, thth, tt = np.meshgrid(x, y, theta, t, indexing='ij')
xx, yy, thth = np.meshgrid(x, y, theta)
vv, ww = np.meshgrid(v, w)
# import pdb; pdb.set_trace()
# Flatten the grid to get a list of 3D points
state_space = np.vstack([xx.ravel(), yy.ravel(), thth.ravel()]).T
control_space = np.vstack([vv.ravel(), ww.ravel()]).T  
# P = compute_transition_probability(control_space, state_space)
# np.save("transition_matrix_100iters_larger_space_larger_prob.npy",P)
# import pdb; pdb.set_trace()
# PI = policy_iteration(control_space, state_space)
# np.save("policy_30iter_PI_50iter_larger_space_larger_prob.npy", PI)

# P = np.load("transition_matrix_100iters.npy")
# print("Testing prob",P[1,2,3,4,5,6,:,:])
# import pdb; pdb.set_trace()
def extract_policy(t, current_error):
    path = 'policy_30iter_PI_50iter_larger_space_larger_prob.npy'
    POLICY = np.load(path)
    # current_state = np.array([1.5       , 0.        , 1.57079633])
    x,y,th = state_metric_to_index(current_error)
    x = np.clip(x, 0, 20)
    y = np.clip(y, 0, 20)
    rule = np.array([i for i in range(40)])
    if np.isin(th, rule) == False:
        length = len(rule)
        th = rule[(th % length)]
    t = (t + 1) % 100
    control = POLICY[t, x, y, th, :]
    return control_index_to_metric(control)


# import pdb; pdb.set_trace()
# compute_transition_probability(control_space, state_space)
# compute_stage_cost(control_space, state_space)
# error_function(0, state_space[:,:3], control_space, noise=True)
# print(state_space[:,:3].shape)
# all_state_indices = state_metric_to_index(state_space, mode="state_space")
# e_x_idx, e_y_idx, e_theta_idx, e_t_idx = all_state_indices[:,0], all_state_indices[:,1], all_state_indices[:,2], all_state_indices[:,3]
# all_control_indices = control_metric_to_index(control_space)  
# print(all_control_indices.shape)
# u_v_idx, u_w_idx = all_control_indices[:,0], all_control_indices[:,1]