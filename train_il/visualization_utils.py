from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import numpy as np
import os
if os.environ.get("RUNNING_IN_LIGHTNING", "1") == "1":
    matplotlib.use("Agg")

from svf.environments.insertion_peg_hole import InsertionPegHole
from mushroom_rl.core import Core, Environment, Agent
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils import VideoRecorder, TorchUtils

from svf.utils.soft_clipping import soft_clip_hard_clip
from svf.network_models import ManifoldSVF
from svf.network_models.rl.actor import PolicyNetworkMSVF, PolicyMSVF, PolicyNetworkMLP
from svf.utils import to_numpy


# from scripts.train_rl.insertion_box_test_msvf import PolicyMSVF
# from ..train_rl.insertion_box_test_msvf import PolicyMSVF

import torch

'''
Load a model with environment specifications

Returns the model and the action indices
'''
def load_model(checkpoint_path, env_id="InsertionBox2D"):
    mdp: InsertionPegHole = Environment.make(env_id, horizon=150)
    mdp.max_steps = 10000
    is_2d = mdp.is_2d
    action_pos_idx = mdp.action_pos_idxs
    action_rot_idx = mdp.action_rot_idxs

    policy_args = dict(
        is_2d=mdp.is_2d,
        pos_goal=mdp.peg_pos_goal,
        rot_goal=mdp.peg_rot_goal,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    # Remove 'model.' prefix from the keys in the state_dict
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model = PolicyNetworkMSVF((1,), (1,), 64, **policy_args)
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()

    return model, action_pos_idx, action_rot_idx, is_2d


def generate_taskspace_quiver_plot(
    checkpoint_path,
    env_id="InsertionBox2D",
    model=None,
    taskspace=None,
    save_path=Path("logs/plots/unnamed.png"),
    action_pos_idx = None,
    action_rot_idx = None,
    num_samples_y = 20,
    num_samples_x = 25,
    is_2d = None,
    panda = False,
):
    if model is not None and (action_pos_idx is None or action_rot_idx is None or is_2d is None):
        raise ValueError("Error: Action indices were missing for the given model")

    if model is None:
        model, action_pos_idx, action_rot_idx, is_2d = load_model(checkpoint_path=checkpoint_path, env_id=env_id)

    if taskspace is None and not panda:
        taskspace = [[-1.0, 1], [0.0, 0.7]]
    elif panda:
        taskspace = [[-0.1, 0.1], [0.0, .17]]

    # Define a grid for visualization
    X, Z = np.meshgrid(
        np.linspace(taskspace[0][0], taskspace[0][1], num_samples_y),  # vertical component
        np.linspace(taskspace[1][0], taskspace[1][1], num_samples_x),  # horizontal component
    )
    Y = np.zeros_like(Z)
    if is_2d:
        # Z is vertical and X is horizontal
        positions = np.stack([Z.ravel(), X.ravel(), Y.ravel()], axis=-1)
    else:
        # Z is vertical component and XY plane is horizontal
        positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    rotation_matrix = np.eye(3).flatten()
    state_ravel = np.concatenate(
        [
            positions,
            np.tile(rotation_matrix, (positions.shape[0], 1)),
        ],
        axis=-1,
    )

    state_ravel_tensor = torch.tensor(state_ravel, dtype=torch.float32)

    # Predict actions
    with torch.no_grad():
        actions = to_numpy(
            model(state_ravel_tensor.to(next(model.parameters()).device))
        )

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    if is_2d:
        dx, dy = actions[..., action_pos_idx[0]], actions[..., action_pos_idx[1]]
    else:
        dy, dx = actions[..., action_pos_idx[0]], actions[..., action_pos_idx[2]]

    epsilon = 1e-8  # Prevent division by zero
    # Create scale factor to compress large values but keep small ones for a better visibility of smaller actions
    scale_factor = np.log1p(np.linalg.norm(np.stack([dx, dy]), axis=0))
    scale_factor[scale_factor == 0] = epsilon
    dx_scaled = dx / scale_factor
    dy_scaled = dy / scale_factor
    ax1.quiver(X, Z, dy_scaled, dx_scaled, angles="xy", scale_units="xy")
    ax1.set_xlim(taskspace[0][0], taskspace[0][1])
    ax1.set_ylim(taskspace[1][0], taskspace[1][1])
    ax1.set_title("Model predictions with log scaling")
    ax1.set_xlabel("X in cm")
    ax1.set_ylabel("Z in cm")
    ax1.grid()
    fig1.savefig(save_path, format="png", dpi=300)

    # Stream plot
    dx_reshape = dx_scaled.reshape(Z.shape)
    dy_reshape = dy_scaled.reshape(X.shape)
    speed = np.sqrt(dx_reshape ** 2 + dy_reshape ** 2)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    stream = ax2.streamplot(X, Z, dy_reshape, dx_reshape, color=speed, cmap='plasma', density=2)
    cbar = fig2.colorbar(stream.lines, ax=ax2)
    cbar.set_label('Speed')
    ax2.set_xlim(taskspace[0][0], taskspace[0][1])
    ax2.set_ylim(taskspace[1][0], taskspace[1][1])
    ax2.set_title("Model predictions with log scaling")
    ax2.set_xlabel("X in cm")
    ax2.set_ylabel("Z in cm")
    fig2.savefig("logs/plots/stream.png", format="png", dpi=300)

    return fig1, fig2


def visualize_2D_quiver_plot(
    checkpoint_path=None, env_id="InsertionBox2D", path=None, traj=None
):
    trajectory_states_l = []
    trajectory_actions_l = []

    # Loads dataset from path
    if path is not None:
        file_list = []
        for file in os.listdir(path):
            if file.endswith(".h5"):
                file_list.append(path + file)

        for file in file_list:
            with h5py.File(file, "r") as h5_file:
                for key in h5_file.keys():
                    if key.startswith("states_") and isinstance(
                        h5_file[key], h5py.Dataset
                    ):
                        trajectory_states_l.append(h5_file[key][:])
                    if key.startswith("actions_") and isinstance(
                        h5_file[key], h5py.Dataset
                    ):
                        trajectory_actions_l.append(h5_file[key][:])

    # Loads samples directly from given trajectories
    elif traj is not None:
        trajectory_states_l.append(traj[0])
        trajectory_actions_l.append(traj[1])

    # Loads Model if checkpoint path is given
    if checkpoint_path is not None:
        model, action_pos_idx, action_rot_idx, is_2d = load_model(checkpoint_path=checkpoint_path, env_id=env_id)

        pred_dx = []
        pred_dy = []

        state_mx = []
        state_my = []
        for state_row in trajectory_states_l:
            for state in state_row:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = model(state_tensor).detach().cpu().numpy().squeeze()
                if is_2d:
                    state_mx.append(state[1])
                    state_my.append(state[0])
                    pred_dx.append(action[action_pos_idx[1]])
                    pred_dy.append(action[action_pos_idx[0]])
                else:
                    state_mx.append(state[0])
                    state_my.append(state[2])
                    pred_dx.append(action[action_pos_idx[0]])
                    pred_dy.append(action[action_pos_idx[2]])

    state_x = []
    state_y = []
    action_dx = []
    action_dy = []

    for state_row, action_row in zip(trajectory_states_l, trajectory_actions_l):
        for state, action in zip(state_row, action_row):
            if checkpoint_path is not None and not is_2d:
                state_x.append(state[0])  # x-coordinate of state
                state_y.append(state[2])  # y-coordinate of state
                action_dx.append(action[0])  # x-component of action
                action_dy.append(action[2])  # y-component of action
            else:
                # in the 2d case, the state[0] value points upwards
                state_x.append(state[1])
                state_y.append(state[0])
                action_dx.append(action[1])
                action_dy.append(action[0])
    change_dx = np.diff(state_x)
    change_dy = np.diff(state_y)

    # Filter states which go from terminal position to the new beginning position
    change_norm = np.sqrt(change_dx**2 + change_dy**2)

    # Choose epsilon big enough to resemble impossible state jumps, here half the depth of the peg hole
    epsilon = 0.2
    valid_indices = np.where(change_norm < epsilon)[0]

    state_x_np = np.array(state_x)
    state_y_np = np.array(state_y)

    filtered_x = state_x_np[valid_indices]
    filtered_y = state_y_np[valid_indices]
    filtered_dx = change_dx[valid_indices]
    filtered_dy = change_dy[valid_indices]

    # Create quiver plot
    plt.figure(figsize=(8, 6))
    plt.quiver(
        state_x,
        state_y,
        action_dx,
        action_dy,
        angles="xy",
        scale_units="xy",
        color="blue",
    )
    plt.quiver(
        filtered_x[:],
        filtered_y[:],
        filtered_dx,
        filtered_dy,
        angles="xy",
        scale_units="xy",
        color="red",
    )
    if checkpoint_path is not None:
        plt.quiver(
            state_mx,
            state_my,
            pred_dx,
            pred_dy,
            angles="xy",
            scale_units="xy",
            color="green",
        )
    plt.title("Quiver Plot of States and Actions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid()
    plt.axis("equal")
    plt.show()


def record_policy_behaviour(
    checkpoint_path,
    env_id,
    num_episodes=5,
    save_vid=True,
    num_features=64,
    vid_dir="logs/vids",
    model_type = "MSVF",
):
    # Create vid directory if not already existing
    os.makedirs(vid_dir, exist_ok=True)

    # Initialize the environment
    mdp: InsertionPegHole = Environment.make(env_id, horizon=150)
    mdp.max_steps = 10000

    if model_type == "MSVF":
        # Parameters for the GaussianTorchPolicy
        policy_args = dict(
            is_2d=mdp.is_2d,
            pos_goal=mdp.peg_pos_goal,
            rot_goal=mdp.peg_rot_goal,
            parse_obs_vec_to_dict=mdp.parse_obs_vec_to_dict,
            is_evaluation = True,
            n_features = num_features,
        )

        # Load and prepare checkpoint state dictionary to load properly
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix from the keys in the state_dict
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        policy = PolicyMSVF(**policy_args)
        policy.policy_network_msvf.load_state_dict(state_dict)

        # Disable gradients for all parameters
        for param in policy.policy_network_msvf.parameters():
            param.requires_grad = False

        policy.policy_network_msvf.to("cpu")
        policy.policy_network_msvf.eval()

    elif model_type == "MLP":
        # Load and prepare checkpoint state dictionary to load properly
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix from the keys in the state_dict
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        policy = PolicyNetworkMLP(
            input_shape=mdp.info.observation_space.shape,
            output_shape=mdp.info.action_space.shape,
            n_features=num_features,
            is_evaluation=False,
        )

        policy.policy_network.load_state_dict(state_dict)

        # Disable gradients for all parameters
        for param in policy.policy_network.parameters():
            param.requires_grad = False

        policy.policy_network.to("cpu")
        policy.policy_network.eval()
    else:
        print("Wrong Model")
        exit()

    # Create the Agent for the Core
    agent = Agent(
        mdp.info,
        policy,
    )

    record_dict = dict(
        recorder_class=VideoRecorder,
        path=vid_dir,
        video_name="recording",
    )

    # Create MushroomRL Core
    core = Core(agent, mdp, record_dictionary=record_dict)

    # Run trials
    dataset = core.evaluate(n_episodes=num_episodes, render=True, record=save_vid)

    states = dataset.state
    actions = dataset.action
    success_number = np.count_nonzero(dataset.absorbing)
    success_rate = success_number / num_episodes

    return dataset, success_rate, states, actions


if __name__ == "__main__":
    # Example usage for how to evaluate a 2D model with given checkpoints from lightning torch
    dataset_path = "dataset/training_2D/"
    model_path_2D = "../../logging/IL_training_2D/2025-04-03_16-46-14/models/il_2D_w_prepos_dataset_1743691930.ckpt"
    model_path_3D = "../../logging/IL_training_3D/2025-04-06_19-59-05/models/il_3D_w_prepos_dataset_1743963537.ckpt"
    model_path_InsertionPandaLShapeRot = "../../logging/IL_training_3D/InsertionLShape3DRot/2025-05-21_17-19-53/models/il_3D_1747842914.ckpt"

    # dataset, success_rate, states, actions = record_policy_behaviour(model_path_3D, num_episodes=1, env_id="InsertionBox3D")
    # visualize_2D_quiver_plot(traj=[states, actions], checkpoint_path=model_path_3D, env_id="InsertionBox3D")
    # visualize_2D_quiver_plot(dataset_path)
    # generate_taskspace_quiver_plot(
    #     model=None, checkpoint_path=model_path_InsertionPandaLShapeRot, taskspace=[[-0.2, 0.2], [0.0, .2]], env_id="InsertionPandaLShapeRot"
    # )
    generate_taskspace_quiver_plot(
        model=None, checkpoint_path=model_path_InsertionPandaLShapeRot, taskspace=[[-1, 1], [0.0, 0.7]],
        env_id="InsertionLShape3DRot"
    )
