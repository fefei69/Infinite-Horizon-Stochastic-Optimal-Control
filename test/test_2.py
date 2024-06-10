import numpy as np

action_dict = {
    0: 'West',
    1: 'East',
    2: 'North',
    3: 'South'
}
actions = [
    np.array([0,-1]),  # West
    np.array([0,1]),   # East
    np.array([-1,0]),  # North
    np.array([1,0])    # South
]

def generate_state_space():
    state_space = []
    for i in range(5):
        for j in range(5):
            state_space.append(np.array([i,j]))
    return state_space

policy = np.zeros((5, 5), dtype=str)
v = np.zeros((5, 5))
state_space = generate_state_space()

for _ in range(300):
    for i in range(5):
        for j in range(5):
            v_candidate = []
            action_value_dict = {}
            
            if i == 0 and j == 1:
                # special state A
                v[i, j] = -10 + 0.9 * v[4, 1]
                continue
            elif i == 0 and j == 3:
                # special state B
                v[i, j] = -5 + 0.9 * v[2, 3]
                continue
            
            for action_id, action in enumerate(actions):
                new_pos = np.array([i, j]) + action
                if not any(np.array_equal(new_pos, arr) for arr in state_space):
                    v_candidate.append(1.0 + 0.9 * v[i, j])
                    action_value_dict[1.0 + 0.9 * v[i, j]] = action_id
                    continue
                
                v_x_prime = v[new_pos[0], new_pos[1]]
                v_candidate.append(0.9 * v_x_prime)
                action_value_dict[0.9 * v_x_prime] = action_id
            
            min_v = min(v_candidate)
            v[i, j] = min_v
            index = v_candidate.index(min_v)
            policy[i, j] = action_dict[action_value_dict[min_v]]

print("Optimal Policy:")
for row in policy:
    print(' '.join(row))
print("\nOptimal Value Function:")
print(v)
