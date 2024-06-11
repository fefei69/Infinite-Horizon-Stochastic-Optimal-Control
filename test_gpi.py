import numpy as np
import utils
from scipy.stats import multivariate_normal
from tqdm import tqdm
def wrap_angle(theta):
    """
    Wraps theta to the range [-pi, pi].
    """
    wrapped_theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return wrapped_theta

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


def state_metric_to_index(metric_state: np.ndarray, mode="time") -> tuple:
    """
    Convert the metric state to grid indices according to your descretization design.
    Args:
        metric_state (np.ndarray): metric state (x, y, theta, t)
    Returns:
        tuple: grid indices
    """
    res = get_resolution()
    if mode == "time":
        m = np.array([-3, -3, -np.pi, 0])
        r = res[:4]
    elif mode == "withou_time":
        # This is mutidimensional version, input is |x| x 7 x 3 x |u| 
        m = np.array([-3, -3, -np.pi])[np.newaxis, np.newaxis, :, np.newaxis]
        r = res[:3][np.newaxis, np.newaxis, :, np.newaxis]
    elif mode == "multi-dim_state":
        # This is mutidimensional version, input is |x| x 4 x |u| 
        m = np.array([-3, -3, -np.pi, 0])[np.newaxis, :, np.newaxis]
        r = res[:4][np.newaxis, :, np.newaxis]
    return np.floor((metric_state-m)/r).astype(int)

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

def get_neighbors(state: np.ndarray) -> tuple:
    # state: (|x|,4,|u|)
    # Resolution of the grid
    res = get_resolution()
    r = res[:3]
    # x,y,theta,t = state
    # state = state[:3]


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
    total_prob = np.sum(probabilities)
    return probabilities/total_prob

def add_noise(mean, samples):
    # mean: |x| x 4 x |u|
    # samples: |x| x 7 x 3 x |u|
    dim_states = mean.shape[0]
    dim_controls = mean.shape[2]
    res = get_resolution()
    r = np.square(res[:3])
    # rescaled sigma to grids unit
    sigma = np.array([[0.04/r[0], 0, 0],
                      [0, 0.04/r[1], 0],
                      [0, 0, 0.004/r[2]]])
    res = get_resolution()
    r = res[:3]
    # import pdb; pdb.set_trace()
    mean, t = mean[:,:3,:], mean[:,3,:]
    samples = np.array(samples)
    probabilities = np.zeros((dim_states, 7, 1, dim_controls), dtype=np.float16)
    for i in range(dim_states):
        for j in range(dim_controls):
            rv = multivariate_normal(mean[i,:,j], sigma)
            probabilities[i,:,:,j] = normalize_prob(rv.pdf(samples[i, :, :, j])).reshape(-1,1)
    full_samples = np.concatenate((samples, probabilities), axis=2)
    # TODO: test reshape
    full_samples.transpose(0, 3, 1, 2)
    # current shape: |x| x 7 x 4 x |u|
    return full_samples.transpose(0, 3, 1, 2) #desired output: |x| x |u| x 7 x 4

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
    # P = np.zeros((100, 21, 21, 40, 6, 11, 7, 4))
    P = np.zeros((100, 11, 11, 10, 6, 11, 7, 4))
    for t in tqdm(range(100)):
        P[t,:,:,:,:,:,:,:] = error_function(t, state_space[:,:3], control_space, noise=True).reshape((11, 11, 10, 6, 11, 7, 4))
    # print("Testing prob",P[1,2,3,4,5,6,:,:])
    # import pdb; pdb.set_trace()
    np.save("transition_matrix_100iters.npy",P)
# Transition matrix
# P_stored = np.load('transition_matrix_10iters.npy')
def compute_stage_cost(control_space, state_space):
    """
    Compute the stage cost L(t, x, y, theta, t, v, w) for all states and controls.
    """
    L = np.zeros(100, 11, 11, 10, 6, 11)

# Discount factor
gamma = 0.95

# Stage cost
L = np.zeros((10,11,11,10)) 

x = np.linspace(-3, 3, 11)
y = np.linspace(-3, 3, 11)
theta = np.linspace(-np.pi, np.pi, 10, endpoint=False)
t = np.linspace(0, 99, 10)

# Create Contol Space
v = np.linspace(0, 1, 6)
w = np.linspace(-1, 1, 11)

# Create the 3D grid using meshgrid
# xx, yy, thth, tt = np.meshgrid(x, y, theta, t, indexing='ij')
xx, yy, thth = np.meshgrid(x, y, theta)
vv, ww = np.meshgrid(v, w)

# Flatten the grid to get a list of 3D points
state_space = np.vstack([xx.ravel(), yy.ravel(), thth.ravel()]).T
control_space = np.vstack([vv.ravel(), ww.ravel()]).T  
# import pdb; pdb.set_trace()
compute_transition_probability(control_space, state_space)
# error_function(0, state_space[:,:3], control_space, noise=True)
# print(state_space[:,:3].shape)
# all_state_indices = state_metric_to_index(state_space)
# e_x_idx, e_y_idx, e_theta_idx, e_t_idx = all_state_indices[:,0], all_state_indices[:,1], all_state_indices[:,2], all_state_indices[:,3]
# all_control_indices = control_metric_to_index(control_space)  
# print(all_control_indices.shape)
# u_v_idx, u_w_idx = all_control_indices[:,0], all_control_indices[:,1]