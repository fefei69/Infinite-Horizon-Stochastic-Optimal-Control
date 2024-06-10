import numpy as np

action_dict = {
    0: 'North',
    1: 'South',
    2: 'East',
    3: 'West'
}
actions = [
    np.array([0,-1]),
    np.array([0,1]),
    np.array([-1,0]),
    np.array([1,0])
]

def generate_state_space():
    state_space = []
    for i in range(5):
        for j in range(5):
            state_space.append(np.array([i,j]))
    return state_space


policy = np.zeros((5, 5),dtype=str)
v = np.zeros((5,5))
state_space = generate_state_space()
for _ in range(3500):
    for i in range(5):
        for j in range(5):
            v_candidate = []
            dict = {}
            if i == 0 and j == 1:
                # special state A
                v[i,j] = -10 + 0.9*v[4,1]
                continue
            elif i == 0 and j == 3: 
                # special state B
                v[i,j] = -4 + 0.9*v[2,3]
                continue
            for action_id, action in enumerate(actions):
                new_pos = np.array([i,j])+action
                if any(np.array_equal(new_pos, arr) for arr in state_space)==False:
                    v_candidate.append(1.0+0.9*v[i,j])
                    dict[1.0+0.9*v[i,j]] = action_id
                    continue
                v_x_prime = v[new_pos[0],new_pos[1]]
                v_candidate.append(0.9*v_x_prime)
                dict[0.9*v_x_prime] = action_id
            min_v = min(v_candidate)
            v[i,j] = min_v
            index = v_candidate.index(min_v)
            policy[i,j] = action_dict[dict[min_v]]
print(policy)
print(v)