import numpy as np
import utils
from scipy.stats import multivariate_normal
from tqdm import tqdm
import ray
# x = np.linspace(-3, 3, 21)
# y = np.linspace(-3, 3, 21)
# theta = np.linspace(-np.pi, np.pi, 40, endpoint=False)
# t = np.linspace(0, 99, 100)
# # Create Contol Space
# v = np.linspace(0, 1, 6)
# w = np.linspace(-1, 1, 11)

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


def state_metric_to_index(metric_state: np.ndarray) -> tuple:
    """
    Convert the metric state to grid indices according to your descretization design.
    Args:
        metric_state (np.ndarray): metric state (x, y, theta, t)
    Returns:
        tuple: grid indices
    """
    m = np.array([-3, -3, -np.pi, 0])
    res = get_resolution()
    r = res[:4]
    return tuple(np.floor((metric_state-m)/r).astype(int))

def state_index_to_metric(state_index: tuple) -> np.ndarray:
    """
    Convert the grid indices to metric state according to your descretization design.
    Args:
        state_index (tuple): grid indices (x, y, theta, t)
    Returns:
        np.ndarray: metric state
    """
    m = np.array([-3, -3, -np.pi, 0])
    res = get_resolution()
    r = res[:4]
    return (state_index)*r+m

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
    return tuple(np.floor((control_metric-m)/r).astype(int))

def control_index_to_metric(control_index:tuple) -> tuple:
    """
    Args:
        v: [N, ] array of indices in the v space
        w: [N, ] array of indices in the w space
    Returns:
        [2, N] array of controls in metric space
    """
    m = np.array([0, -1])
    res = get_resolution()
    r = res[4:]
    return (control_index)*r+m

def get_neighbors(state: np.ndarray) -> tuple:
    # Resolution of the grid
    res = get_resolution()
    r = res[:3]
    x,y,theta,t = state
    state = state[:3]
    neighbors = [state, 
                 state + np.array([r[0], 0, 0]),
                 state + np.array([-r[0], 0, 0]),
                 state + np.array([0, r[1], 0]),
                 state + np.array([0, -r[1], 0]),
                 state + np.array([0, 0, r[2]]),
                 state + np.array([0, 0, -r[2]]),
    ]
    # neighbors index 
    neighbors_index = []
    for i in range(len(neighbors)):
        neighbors_index.append(state_metric_to_index(np.append(neighbors[i],t)))
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
    # diag_sigma = np.array([0.04, 0.04, 0.004])
    # expanded_diag = np.zeros((6,6))
    # import pdb; pdb.set_trace()
    # np.fill_diagonal(expanded_diag, diag_sigma)
    # expanded_diag[:3] = diag_sigma
    sigma = np.array([[0.04, 0, 0],
                      [0, 0.04, 0],
                      [0, 0, 0.004]])
    # sigma = np.square(sigma)
    
    mean, t = mean[:3], mean[3]
    samples = np.array(samples)
    # meam = np.array(mean)
    # multi_mean = np.append(mean, mean)
    # more_sigma = np.stack(sigma, sigma)    
    # import pdb; pdb.set_trace()
    # Create a multivariate normal distribution 
    rv = multivariate_normal(mean, sigma)
    # import pdb; pdb.set_trace()
    # Compute the probabilities for each state
    probabilities = rv.pdf(samples[:, :3]) # slice samples 8x4 to 8x3
    # Normalize the probabilities
    return normalize_prob(probabilities) 

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
    traj = utils.lissajous
    curr_ref = traj(t)
    next_ref = traj(t+1)

    curr_ref = np.array(curr_ref)
    next_ref = np.array(next_ref)
    time_interval = utils.time_step
    G_et = np.array([[time_interval * np.cos(curr_error[2]+curr_ref[2]), 0],
                     [time_interval * np.sin(curr_error[2]), 0],
                     [0, time_interval]])
    # Compute the next error at time t+1
    next_error = curr_error + G_et @ u_t + (curr_ref-next_ref)
    # Wrap the angle
    next_error[2] = wrap_angle(next_error[2])
    next_error = np.append(next_error, t+1)
    # Get the 7 neighbors of the next error
    neighbors, neighbors_index = get_neighbors(next_error)
    # Consider the noise and get the probabilities of transitioning to each neighbor
    next_error = state_metric_to_index(next_error)
    probabilities = add_noise(next_error, neighbors_index)
    if noise:
        return neighbors, neighbors_index, probabilities 
    else:
        print("Should add noise or implement the noiseless version")
        raise NotImplementedError

def compute_transition_probability(control_space, state_space):
    # Initialize the transition probability matrix P (t, x, y, theta, v, w, 7, 4)
    # P = np.zeros((100, 21, 21, 40, 6, 11, 7, 4))
    P = np.zeros((10, 11, 11, 10, 6, 11, 7, 4))
    for t in range(10):
        for u in tqdm(control_space):
            for e in state_space:
                neighbors, neighbors_index, probabilities = error_function(t, e[:3], u, noise=True)
                e_x_idx, e_y_idx, e_theta_idx, e_t_idx = state_metric_to_index(e)
                u_v_idx, u_w_idx = control_metric_to_index(u)
                # Assign the probabilities to neighbors indices
                for i in range(7):
                    neighbors_state = neighbors[i]
                    P[t, e_x_idx, e_y_idx, e_theta_idx, u_v_idx, u_w_idx, i] = np.append(neighbors_state, probabilities[i])
    return P


# ray.init()

# @ray.remote
# def worker(t, u, state_space_chunk):
#     results = []
#     for e in state_space_chunk:
#         neighbors, neighbors_index, probabilities = error_function(t, e[:3], u, noise=True)
#         e_x_idx, e_y_idx, e_theta_idx, e_t_idx = state_metric_to_index(e)
#         u_v_idx, u_w_idx = control_metric_to_index(u)
#         for i in range(7):
#             neighbors_state = neighbors[i]
#             results.append((t, e_x_idx, e_y_idx, e_theta_idx, u_v_idx, u_w_idx, i, np.append(neighbors_state, probabilities[i])))
#     return results

# def chunk_state_space(state_space, num_chunks):
#     chunk_size = len(state_space) // num_chunks
#     return [state_space[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

# def compute_transition_probability_parrallel(control_space, state_space, num_workers=4):
#     P = np.zeros((10, 11, 11, 10, 6, 11, 7, 4))
#     jobs = []

#     state_space_chunks = chunk_state_space(state_space, num_workers)
    
#     for t in range(1):
#         for u in tqdm(control_space):
#             for state_space_chunk in state_space_chunks:
#                 jobs.append(worker.remote(t, u, state_space_chunk))
    
#     num_jobs = len(jobs)
#     print("Number of jobs", num_jobs)
#     # pbar = tqdm(total = num_jobs)
#     # while jobs:
#     #     done_ids, jobs = ray.wait(jobs, num_returns=1)
#     #     local_P = ray.get(done_ids[0])
#     #     for t, e_x_idx, e_y_idx, e_theta_idx, u_v_idx, u_w_idx, i, values in local_P:
#     #         P[t, e_x_idx, e_y_idx, e_theta_idx, u_v_idx, u_w_idx, i] = values
#     #     pbar.update(1)
#         # P += local_P

#     results = ray.get(jobs)
#     print("Aggregating Results")
#     for result in tqdm(results):
#         for t, e_x_idx, e_y_idx, e_theta_idx, u_v_idx, u_w_idx, i, values in result:
#             P[t, e_x_idx, e_y_idx, e_theta_idx, u_v_idx, u_w_idx, i] = values
    
#     ray.shutdown()
#     return P

if __name__ == '__main__':
    
    print("Test Discretizing states")
    # Create State Space
    x = np.linspace(-3, 3, 11)
    y = np.linspace(-3, 3, 11)
    theta = np.linspace(-np.pi, np.pi, 10, endpoint=False)
    t = np.linspace(0, 99, 10)
    # Create Contol Space
    v = np.linspace(0, 1, 6)
    w = np.linspace(-1, 1, 11)
    
    # Create the 3D grid using meshgrid
    xx, yy, thth, tt = np.meshgrid(x, y, theta, t, indexing='ij')
    vv, ww = np.meshgrid(v, w)

    # Flatten the grid to get a list of 3D points
    state_space = np.vstack([xx.ravel(), yy.ravel(), thth.ravel(), tt.ravel()]).T
    control_space = np.vstack([vv.ravel(), ww.ravel()]).T   
    
    # print("Test Discretizing state space.........................")
    # print("There are ", state_space.shape, "states")
    # random_state = state_space[np.random.choice(1000, 1)].squeeze()
    # print("Random state in the state_space:", random_state)  
    # print("Convert State to index", state_metric_to_index(random_state))
    # print("Convert index back to State", state_index_to_metric(state_metric_to_index(random_state)))
    # print("\n")

    # print("First state:", state_space[0])
    # print("Convert First State to index", state_metric_to_index(state_space[0]))
    # print("Convert index back to State", state_index_to_metric(state_metric_to_index(state_space[0])))
    # print("\n")

    # print("Last state:", state_space[-1])
    # print("Convert Last State to index", state_metric_to_index(state_space[-1]))
    # print("Convert index back to State", state_index_to_metric(state_metric_to_index(state_space[-1])))
    # print("\n")

    # print("Test Discretizing control space.........................")
    # print("There are ", control_space.shape, "controls")

    # random_control = control_space[np.random.choice(100, 1)].squeeze()
    # print("Random control in the control_space:", random_control)
    # print("Convert Control to index", control_metric_to_index(random_control))
    # print("Convert index back to Control", control_index_to_metric(control_metric_to_index(random_control)))
    # print("\n")

    # print("First control:", control_space[0])
    # print("Convert First Control to index", control_metric_to_index(control_space[0]))
    # print("Convert index back to Control", control_index_to_metric(control_metric_to_index(control_space[0])))
    # print("\n")

    # print("Last control:", control_space[-1])
    # print("Convert Last Control to index", control_metric_to_index(control_space[-1]))
    # print("Convert index back to Control", control_index_to_metric(control_metric_to_index(control_space[-1])))

    # print("Test get_neighbors")
    # neighbors_list, neighbors_index = get_neighbors(random_state)
    # print("Neighbors of random state", random_state)
    # print(neighbors_list)
    # print(neighbors_index)

    # print("Test transition probability")
    # print("Random state", random_state)
    # print("Random control", random_control)

    # t = random_state[3]
    # neighbors, neighbors_index, probabilities = error_function(t, random_state[:3], random_control, noise=True)
    # print("Error with noise at t+1", neighbors)
    # print("Indices of neighbors", neighbors_index)
    # print("Probabilities", probabilities)
    # print("Sum of probabilities", np.sum(probabilities))


    #TODO: wrap around the angles and check if it's normal to have 1 prob in the neighbors
    print("Test transition probability")  
    e = state_space[0]
    u = control_space[0]
    e_x_idx, e_y_idx, e_theta_idx, e_t_idx = state_metric_to_index(e)
    u_v_idx, u_w_idx = control_metric_to_index(u)
    print("State id", e_x_idx, e_y_idx, e_theta_idx, e_t_idx)   
    print("Control id", u_v_idx, u_w_idx)
    P = compute_transition_probability(control_space, state_space)
    print(P[0, e_x_idx, e_y_idx, e_theta_idx, u_v_idx, u_w_idx])
    # np.save("transition_matrix_iter1_corrected.npy", P)
    # import pdb; pdb.set_trace()