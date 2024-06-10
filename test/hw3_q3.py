import numpy as np

# Define grid dimensions
grid_size = 5

# Define special states and their transitions
special_states = {
    (0, 1): (-10, (4, 1)),  # A -> A'
    (0, 3): (-5, (2, 3))    # B -> B'
}

# Discount factor
gamma = 0.9

# Initialize value function
V = np.zeros((grid_size, grid_size))

# Define the possible actions (north, south, east, west)
actions = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1)
}

# Reward and cost structure
default_cost = -1

def get_next_state_and_cost(state, action):
    if state in special_states:
        return special_states[state]
    
    next_state = (state[0] + action[0], state[1] + action[1])
    
    if next_state[0] < 0 or next_state[0] >= grid_size or next_state[1] < 0 or next_state[1] >= grid_size:
        return state, default_cost  # No movement and default cost if out of bounds
    return next_state, 0

def value_iteration(V, gamma, threshold=1e-4):
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for i in range(grid_size):
            for j in range(grid_size):
                v = V[i, j]
                max_value = float('-inf')
                for action in actions.values():
                    next_state, cost = get_next_state_and_cost((i, j), action)
                    max_value = max(max_value, cost + gamma * V[next_state])
                
                new_V[i, j] = max_value
                delta = max(delta, abs(v - new_V[i, j]))
        
        V = new_V
        
        if delta < threshold:
            break
    
    return V

def extract_policy(V, gamma):
    policy = np.full((grid_size, grid_size), ' ')
    
    for i in range(grid_size):
        for j in range(grid_size):
            max_value = float('-inf')
            best_action = ' '
            for action_key, action_value in actions.items():
                next_state, cost = get_next_state_and_cost((i, j), action_value)
                value = cost + gamma * V[next_state]
                if value > max_value:
                    max_value = value
                    best_action = action_key
            
            policy[i, j] = best_action
    
    return policy

# Perform value iteration
V_optimal = value_iteration(V, gamma)

# Extract optimal policy
optimal_policy = extract_policy(V_optimal, gamma)

# Display results
print("Optimal Value Function:")
print(V_optimal)

print("\nOptimal Policy:")
for row in optimal_policy:
    print(' '.join(row))
