import numpy as np

action_dict = {
    0: 'West',
    1: 'East',
    2: 'North',
    3: 'South'
}
actions = [
    np.array([0, -1]),  # West
    np.array([0, 1]),   # East
    np.array([-1, 0]),  # North
    np.array([1, 0])    # South
]

def generate_state_space():
    state_space = []
    for i in range(5):
        for j in range(5):
            state_space.append(np.array([i, j]))
    return state_space

def q_value_iteration(gamma=0.9, theta=1e-4):
    state_space = generate_state_space()
    q_values = np.zeros((5, 5, 4))  # Q-values for each state-action pair
    v = np.zeros((5, 5))  # Value function
    policy = np.full((5, 5), ' ')

    while True:
        delta = 0
        for i in range(5):
            for j in range(5):
                for action_id, action in enumerate(actions):
                    old_q_value = q_values[i, j, action_id]
                    new_pos = np.array([i, j]) + action
                    if (i, j) == (0, 1):
                        q_values[i, j, action_id] = -10 + gamma * v[4, 1]
                    elif (i, j) == (0, 3):
                        q_values[i, j, action_id] = -5 + gamma * v[2, 3]
                    elif not any(np.array_equal(new_pos, arr) for arr in state_space):
                        q_values[i, j, action_id] = 1.0 + gamma * v[i, j]
                    else:
                        q_values[i, j, action_id] = gamma * v[new_pos[0], new_pos[1]]
                    delta = max(delta, abs(old_q_value - q_values[i, j, action_id]))
        
        for i in range(5):
            for j in range(5):
                v[i, j] = np.min(q_values[i, j])
        
        if delta < theta:
            break

    for i in range(5):
        for j in range(5):
            best_action_id = np.argmin(q_values[i, j])
            policy[i, j] = action_dict[best_action_id]

    return policy, v

# Perform Q-value iteration
optimal_policy, optimal_v = q_value_iteration()

# Display results
print("Optimal Policy:")
for row in optimal_policy:
    print(' '.join(row))
print("\nOptimal Value Function:")
print(optimal_v)
