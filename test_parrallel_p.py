import numpy as np
import ray
from tqdm import tqdm
from test_gpi import error_function, state_metric_to_index, control_metric_to_index

ray.init(num_cpus=4)

@ray.remote
def compute_transition_probability_single(t, control_space, state_space):
    P_partial = np.zeros((21, 21, 40, 6, 11, 7, 4))
    for u in tqdm(control_space):
        for e in tqdm(state_space):
            neighbors, neighbors_index, probabilities = error_function(t, e[:3], u, noise=True)
            e_x_idx, e_y_idx, e_theta_idx, e_t_idx = state_metric_to_index(e)
            u_v_idx, u_w_idx = control_metric_to_index(u)
            for i in range(7):
                neighbors_state = neighbors[i]
                P_partial[e_x_idx, e_y_idx, e_theta_idx, u_v_idx, u_w_idx, i] = np.append(neighbors_state, probabilities[i])
    return P_partial

def compute_transition_probability(control_space, state_space):
    # Initialize the transition probability matrix P (t, x, y, theta, v, w, 7, 4)
    P = np.zeros((100, 21, 21, 40, 6, 11, 7, 4))
    tasks = [compute_transition_probability_single.remote(t, control_space, state_space) for t in range(1)]
    results = ray.get(tasks)
    
    for t, P_partial in enumerate(results):
        P[t] = P_partial
    
    return P


print("Test Discretizing states")
# Create State Space
x = np.linspace(-3, 3, 21)
y = np.linspace(-3, 3, 21)
theta = np.linspace(-np.pi, np.pi, 40, endpoint=False)
t = np.linspace(0, 99, 100)
# Create Contol Space
v = np.linspace(0, 1, 6)
w = np.linspace(-1, 1, 11)

# Create the 3D grid using meshgrid
xx, yy, thth, tt = np.meshgrid(x, y, theta, t, indexing='ij')
vv, ww = np.meshgrid(v, w)

# Flatten the grid to get a list of 3D points
state_space = np.vstack([xx.ravel(), yy.ravel(), thth.ravel(), tt.ravel()]).T
control_space = np.vstack([vv.ravel(), ww.ravel()]).T  
print("Test transition probability")  
e = state_space[0]
u = control_space[0]
e_x_idx, e_y_idx, e_theta_idx, e_t_idx = state_metric_to_index(e)
u_v_idx, u_w_idx = control_metric_to_index(u)
print("State id", e_x_idx, e_y_idx, e_theta_idx, e_t_idx)   
print("Control id", u_v_idx, u_w_idx)
P = compute_transition_probability(control_space, state_space)
print(P[0, e_x_idx, e_y_idx, e_theta_idx, u_v_idx, u_w_idx]) 