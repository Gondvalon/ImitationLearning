import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from experiment_launcher.utils import fix_random_seed
from mushroom_rl.core import Core, Environment, Agent

import pybullet as p
from svf.environments.insertion_peg_hole import InsertionPegHole

import h5py
import time


def save_trajectories_to_h5(
    trajectory_observations_l, trajectory_actions_l, save_path, smooth_trajectory
):
    def orthonormalize(matrix):
        """
        Ensure a 3x3 matrix is orthonormal via SVD.
        """
        U, _, Vt = np.linalg.svd(matrix)
        return U @ Vt

    # Requires the trajectories to be lists of lists of np.ndarray
    if smooth_trajectory:
        # Parameters for smoothing
        window_length = 5
        polyorder = 2

        smoothed_traj_observation_l = []
        smoothed_traj_action_l = []

        # Smooth all recorded trajectories
        for i in range(len(trajectory_observations_l)):
            states = np.stack(trajectory_observations_l[i])
            actions = np.stack(trajectory_actions_l[i])

            # Seperate the position vector and the rotation matrix for smoothing
            positions = states[:, :3]
            rotations = states[:, 3:]

            # Smooth trajectory data
            smoothed_positions = savgol_filter(
                positions, window_length=window_length, polyorder=polyorder, axis=0
            )
            smoothed_rotations = savgol_filter(
                rotations, window_length=window_length, polyorder=polyorder, axis=0
            )
            smoothed_actions = savgol_filter(
                actions, window_length=window_length, polyorder=polyorder, axis=0
            )

            # Re-orthonormalize the rotation matrices
            orthonormalized_rotations = []
            for rotation in smoothed_rotations:
                rotation_matrix = rotation.reshape(3, 3)  # Convert to 3x3
                orthonormalized_rotations.append(
                    orthonormalize(rotation_matrix).flatten()
                )

            orthonormalized_rotations = np.array(orthonormalized_rotations)

            # Combine positions and Rotations back to states
            smoothed_state_matrix = np.hstack(
                [smoothed_positions, orthonormalized_rotations]
            )

            # Recreate the lists of states and actions
            smoothed_states = [state for state in smoothed_state_matrix]
            smoothed_actions = [action for action in smoothed_actions]

            # Recreate the list of trajectories
            smoothed_traj_observation_l.append(smoothed_states)
            smoothed_traj_action_l.append(smoothed_actions)

    # Ensure saving structure is existing
    dataset_saving_path = f"{save_path}/smoothed/"
    try:
        # Create the folder(s) if they don't exist
        os.makedirs(dataset_saving_path, exist_ok=True)
        print(f"Folder '{dataset_saving_path}' is ready!")
    except Exception as e:
        print(f"Creating the dir: {dataset_saving_path} threw an error")

    with h5py.File(f"{save_path}/trajectories_{int(time.time())}.h5", "w") as h5file:
        for i in range(len(trajectory_observations_l)):
            h5file.create_dataset(f"states_{i}", data=trajectory_observations_l[i])
            h5file.create_dataset(f"actions_{i}", data=trajectory_actions_l[i])

    if smooth_trajectory:
        with h5py.File(
            f"{save_path}/smoothed/trajectories_{int(time.time())}.h5", "w"
        ) as h5file:
            for i in range(len(smoothed_traj_observation_l)):
                h5file.create_dataset(
                    f"states_{i}", data=smoothed_traj_observation_l[i]
                )
                h5file.create_dataset(f"actions_{i}", data=smoothed_traj_action_l[i])


def record_env(
    record_trajectory,
    smooth_trajectory,
    env_id="InsertionBox2D",
    is_L_shape=False,
    panda=False,
    max_trajectories=50,
    horizon=200,
    peg_init_pos_bounds=None,
    debug=True,
):
    # Sigmoid scaling within limits
    def scale_sigmoid(error_vector, clip_limit, min_limit, tolerance, k=3.0):
        sign = np.sign(error_vector)
        abs_error = np.abs(error_vector)

        # Apply dead zone: Set small errors to zero
        abs_scaled = np.where(
            abs_error < tolerance,
            0.0, # No action within tolerance
            min_limit + (clip_limit - min_limit) / (1 + np.exp(-k * abs_error)),
        )

        return sign * abs_scaled

    # Returns Action for the environment depending on the given stage and observation
    def get_action(
            obs,
            pos_stages,
            rot_stages,
            current_stage_idx,
            is_2d=True,
            clip_limit=0.30,
            min_limit=0.01,
            pos_tolerance=0.005,
            rot_tolerance=0.05,
            k=3.0,
    ):
        current_pos = obs[:3]
        current_rot = obs[3:12].reshape(3, 3)

        pos_goal = pos_stages[current_stage_idx]
        rot_goal = rot_stages[current_stage_idx]

        # Compute rotation error
        rot_error_matrix = rot_goal @ current_rot.T
        raw_rot_error = R.from_matrix(rot_error_matrix).as_rotvec()

        if is_2d:
            rot_error = np.array([raw_rot_error[2]])
        else:
            rot_error = raw_rot_error

        # Compute position error
        if is_2d:
            pos_error = (pos_goal - current_pos)[0:2]
            pos_done = np.all(np.abs(pos_error) < pos_tolerance)
            rot_done = np.abs(rot_error).max() < rot_tolerance
        else:
            pos_error = pos_goal - current_pos
            pos_done = np.all(np.abs(pos_error) < pos_tolerance)
            rot_done = np.all(np.abs(rot_error) < rot_tolerance)

        stage_complete = pos_done and rot_done

        # Scale errors
        scaled_pos_error = scale_sigmoid(pos_error, clip_limit, min_limit, pos_tolerance, k)

        # Combine and clip
        action = np.hstack((scaled_pos_error, rot_error))
        action = np.clip(action, -clip_limit, clip_limit)

        # Enforce minimum magnitude
        action = np.where((action > 0) & (action < min_limit), min_limit, action)
        action = np.where((action < 0) & (action > -min_limit), -min_limit, action)

        return action, stage_complete

    fix_random_seed(np.random.randint(0, 1000000000))

    if peg_init_pos_bounds is not None:
        env: InsertionPegHole = Environment.make(env_id, debug_gui=debug, peg_init_pos_bounds=peg_init_pos_bounds)
    else:
        env: InsertionPegHole = Environment.make(env_id, debug_gui=debug)

    is_2d = env.is_2d

    save_location = f"dataset/{env_id}"

    obs = env.reset()

    # In 3D collisions are deactivated. Not necessary if waypoints/stages are set properly
    if not is_2d:
        for body_id in range(p.getNumBodies()):
            p.setCollisionFilterGroupMask(body_id, -1, 0, 0)

    # Initialize environment and trajectories
    trajectory_observations_l = []
    trajectory_actions_l = []
    obs_l = [obs[0]]
    action_l = []

    # Set up the stages for the different scenarios as waypoints
    pos_goal = env.peg_pos_goal
    rot_goal = env.peg_rot_goal.as_matrix()
    rot_stages = [
        rot_goal,  # Maintain orientation in this example
        rot_goal,
    ]
    if is_2d:
        pos_stages = [
            pos_goal + np.array([0.3, 0.0, 0.0]),  # Above hole
            pos_goal,  # Insert
        ]
    elif not is_2d and is_L_shape:
        pos_stages = [
            pos_goal + np.array([0.0, -0.15, 0.4]),  # Above hole
            pos_goal + np.array([0.0, -0.15, 0.0]),
            pos_goal,  # Insert
            pos_goal
        ]
        rot_stages = [
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # Maintain orientation in this example
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            rot_goal,
        ]
    elif not is_2d and panda:
        pos_stages = [
            pos_goal + np.array([-0.005, -0.0275, 0.09]),  # Above hole
            pos_goal + np.array([-0.005, -0.0275, 0.0]),
            pos_goal,  # Insert
            pos_goal
        ]
        rot_stages = [
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # Maintain orientation in this example
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            rot_goal,
        ]
    else:
        pos_stages = [
            pos_goal + np.array([0.0, 0.0, 0.4]),  # Above hole
            pos_goal,  # Insert

        ]

    # Initialize loop variables
    i = 0
    steps = 0
    stage_idx = 0

    while i < max_trajectories:
        # Actively reset environment via X
        keys = p.getKeyboardEvents()
        if ord("x") in keys and keys[ord("x")] & p.KEY_IS_DOWN and debug:
            obs = env.reset()
            i = i + 1
            steps = 0
            stage_idx = 0

            obs_l = [obs[0]]
            action_l = []
            print("Reset env")

        if debug:
            env.render()

        action, stage_complete = get_action(
            obs_l[-1],
            pos_stages,
            rot_stages,
            stage_idx,
            is_2d=is_2d,
        )

        # Check current stage
        if stage_complete and stage_idx < len(pos_stages) - 1:
            stage_idx += 1

        # Filter out zero actions
        if np.abs(action).sum() != 0.0:
            obs, reward, done, info = env.step(action)
            action_l.append(action)

            if not done and steps < horizon:
                obs_l.append(obs)
                steps = steps + 1
            elif not done and steps == horizon:
                # Unfinished Trajectories will get discarded
                obs = env.reset()
                i = i + 1
                steps = 0
                stage_idx = 0

                obs_l = [obs[0]]
                action_l = []
                print("Reset env")
            else:
                # Save trajectories and reset environment
                trajectory_observations_l.append(np.array(obs_l))
                trajectory_actions_l.append(np.array(action_l))

                obs = env.reset()
                if not is_2d:
                    for body_id in range(p.getNumBodies()):
                        p.setCollisionFilterGroupMask(body_id, -1, 0, 0)

                obs_l = [obs[0]]
                action_l = []
                i = i + 1
                print(f"{i} successful trajectories")
                steps = 0
                stage_idx = 0

    if record_trajectory:
        save_trajectories_to_h5(
            trajectory_observations_l,
            trajectory_actions_l,
            save_location,
            smooth_trajectory=smooth_trajectory,
        )

    print("Exiting...")


def generate_default_pairs(
    record_trajectory, sample_grid_size, env_id="InsertionBox2D", taskspace=None
):
    if taskspace is None:
        taskspace = [[-0.5, 0.5], [0.0, 0.7]]
    save_location = "dataset/training_2D/debug_data"

    env: InsertionPegHole = Environment.make(env_id, debug_gui=False)

    pos_goal = env.peg_pos_goal
    rot_goal = env.peg_rot_goal.as_matrix()
    flat_rot_goal = rot_goal.flatten()

    # Create lists for dataset consistency
    trajectory_observations_l = []
    trajectory_actions_l = []

    obs_l = []
    action_l = []

    x_bounds, y_bounds = taskspace

    x_vals = np.linspace(x_bounds[0], x_bounds[1], sample_grid_size[0])
    y_vals = np.linspace(y_bounds[0], y_bounds[1], sample_grid_size[1])

    for x in x_vals:
        for y in y_vals:
            # Current state is the position (x, y)
            state = np.array(
                [x, y, 0]
            )  # Ignore rotation for now (rotation is 0,0,0 in 2D)

            # Compute the direction from current position (x, y) to goal (goal_pos)
            direction = pos_goal - state
            delta_x, delta_y = direction[0], direction[1]

            # Normalize delta_x, delta_y
            dist = np.linalg.norm([delta_x, delta_y])  # Get distance to goal
            if dist != 0:
                delta_x /= dist
                delta_y /= dist

            # Delta for rotation (delta_phi) is ignored in this case since you said you're ignoring rotation
            delta_phi = 0  # No rotation change for now

            # Action is [dx, dy, delta_phi]
            action = np.array([delta_x, delta_y, delta_phi])

            # Append state-action pair to the list
            full_state = np.concatenate((state, flat_rot_goal))
            obs_l.append(full_state)
            action_l.append(action)

    trajectory_observations_l.append(np.array(obs_l))
    trajectory_actions_l.append(np.array(action_l))

    if record_trajectory:
        save_trajectories_to_h5(
            trajectory_observations_l,
            trajectory_actions_l,
            save_location,
            smooth_trajectory=False,
        )


if __name__ == "__main__":
    COLLECT_DEFAULT = False

    RECORD_TRAJECTORY = True
    SMOOTH_TRAJECTORY = True
    DEBUG = True

    # available envs: InsertionBox2D, InsertionBox3D, InsertionLShape3DRot, InsertionPandaLShapeRot
    env_id = "InsertionPandaLShapeRot"
    max_trajectories = 20

    if not COLLECT_DEFAULT:
        #ensure that the prepositions for the recording fit the env
        record_env(
            env_id=env_id,
            max_trajectories=max_trajectories,
            record_trajectory=RECORD_TRAJECTORY,
            smooth_trajectory=SMOOTH_TRAJECTORY,
            peg_init_pos_bounds=None,
            # peg_init_pos_bounds=[[0.4, -0.7, -0.7], [0.7, 0.7, 0.7]], # for 3D LShape env
            # peg_init_pos_bounds=[[0.00, 0.00, 0.10], [0.09, 0.09, 0.17]],
            debug=DEBUG,
            is_L_shape=True if env_id == "InsertionLShape3DRot" else False,
            panda = True if env_id == "InsertionPandaLShapeRot" else False,
        )
    elif COLLECT_DEFAULT:
        generate_default_pairs(
            env_id="InsertionBox2D",
            record_trajectory=RECORD_TRAJECTORY,
            sample_grid_size=[80, 60],
        )
