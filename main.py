from time import time
import numpy as np
import utils
import argparse

from cec import CEC
from gpi_larger_state_space import extract_policy
def main(control_method):
    # Obstacles in the environment
    obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
    # Params
    traj = utils.lissajous
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    cur_iter = 0
    # control intial guess
    time_horizon = 5
    # Main loop
    while cur_iter * utils.time_step < utils.sim_time:
        t1 = time()
        # Get reference state
        cur_time = cur_iter * utils.time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        # control = utils.simple_controller(cur_state, cur_ref)
        # import pdb; pdb.set_trace()
        control = CEC(cur_iter, cur_state, cur_ref, time_horizon)
        # control = extract_policy(cur_iter, cur_state-cur_ref)     
        # control = control[:2]
        if control_method == 'simple':
            control = utils.simple_controller(cur_state, cur_ref)
        elif control_method == 'cec':
            control = CEC(cur_iter, cur_state, cur_ref, time_horizon)
        elif control_method == 'gpi':
            control = extract_policy(cur_iter, cur_state - cur_ref)
        else:
            raise ValueError("Unknown control method")
        print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = utils.time()
        print("Iteration", cur_iter)
        print(t2 - t1)
        times.append(t2 - t1)
        cur_err = cur_state - cur_ref
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans = error_trans + np.linalg.norm(cur_err[:2])
        error_rot = error_rot + np.abs(cur_err[2])
        print(cur_err, error_trans, error_rot)
        print("======================")
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print("\n\n")
    print("Total time: ", main_loop_time - main_loop, "s")
    print("Average iteration time: ", np.array(times).mean() * 1000, "ms")
    print("Final error_trains: ", error_trans)
    print("Final error_rot: ", error_rot)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, save=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control method selection')
    parser.add_argument('--control_method', type=str, choices=['simple', 'cec', 'gpi'], required=True, help='Control method to use')
    args = parser.parse_args()
    main(args.control_method)

