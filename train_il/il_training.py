from pathlib import Path
import ffmpeg

import wandb

import torch
import torch.nn as nn

import pytorch_lightning as L

import time
import os
import pdb

import matplotlib.pyplot as plt

import pybullet as p

from mushroom_rl.core import Core, Environment, Agent
from svf.environments.insertion_peg_hole import InsertionPegHole

import numpy as np

from svf.network_models.rl.actor import PolicyNetworkMSVF, PolicyNetwork

from scipy.spatial.transform import Rotation as R

from svf.utils import to_torch, to_numpy
from scripts.logging_utils import return_most_recent_file, find_git_root_from_file
from visualization_utils import generate_taskspace_quiver_plot, record_policy_behaviour


class ILTrainingLightning(L.LightningModule):
    def __init__(
        self,
        env_id,
        checkpoint_location,
        logging_dir,
        checkpoint_filename_prefix,
        pos_goal=None,
        rot_goal=None,
        learning_rate=0.001,
        pos_action_alpha=1.0,
        rot_action_alpha=1.0,
        num_trials=10,
        num_features=256,
        max_env_steps=200,
        evaluation_period=10,
        num_rec_episodes= 10,
        record_final_video=True,
        debug=False,
        trainable_network="MSVF",
        **kwargs,
    ):
        super().__init__()

        self.debug = debug
        self.learning_rate = learning_rate
        self.alpha_pos = pos_action_alpha
        self.alpha_rot = rot_action_alpha

        self.env_id = env_id
        self.env: InsertionPegHole = Environment.make(self.env_id, debug_gui=False)
        self.is_2D = self.env.is_2d

        self.action_pos_idx = self.env.action_pos_idxs
        self.action_rot_idx = self.env.action_rot_idxs

        self.num_trials = num_trials
        self.max_env_steps = max_env_steps
        self.evaluation_period = evaluation_period

        self.logging_dir = Path(find_git_root_from_file(__file__), logging_dir)
        print(f"Logging: {self.logging_dir}")
        self.checkpoint_location = checkpoint_location
        self.checkpoint_filename_prefix = checkpoint_filename_prefix

        self.record_final_video = record_final_video
        self.num_rec_episodes = num_rec_episodes

        self.trainable_network = trainable_network

        if pos_goal is None:
            self.pos_goal = [0.0, 0.0, 0.0]
        else:
            self.pos_goal = pos_goal

        if rot_goal is None:
            self.rot_goal = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        else:
            self.rot_goal = rot_goal

        self.num_features = num_features
        # model
        if self.trainable_network == "MSVF":
            self.model = PolicyNetworkMSVF(
                (1,),
                (1,),
                num_features,
                is_2d=self.is_2D,
                pos_goal=self.pos_goal,
                rot_goal=R.from_matrix(self.rot_goal),
            )
        elif self.trainable_network == "MLP":
            self.model = PolicyNetwork(
                input_shape=self.env.info.observation_space.shape,
                output_shape=self.env.info.action_space.shape,
                n_features=num_features
            )

        self.loss = nn.MSELoss()

        # extra folders
        self.plots_dir = self.logging_dir / "plots"
        os.makedirs(self.plots_dir, exist_ok=True)

    def training_step(self, samples_batch, batch_idx):
        # input data
        state = samples_batch["state"].to(self.device)
        action = samples_batch["action"].to(self.device)

        model_out = self.model(state)

        # position velocity loss (dx, dy)
        pos_loss = self.loss(model_out[:, self.action_pos_idx], action[:, self.action_pos_idx])

        # angular velocity loss (dtheta)
        rot_loss = self.loss(model_out[:, self.action_rot_idx], action[:, self.action_rot_idx])

        total_loss = self.alpha_pos * pos_loss + self.alpha_rot * rot_loss

        # Log training loss
        self.log(
            "training_pos_loss",
            pos_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "training_rot_loss",
            rot_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "training_loss",
            total_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    # Safety debug point if NaN values appear
    def on_before_optimizer_step(self, optimizer):
        if self.debug:
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"❌ NaN in gradient of {name}")
                    pdb.set_trace()

    def validation_step(self, sampled_batched, batch_idx):
        # Input data
        state = sampled_batched["state"].to(self.device)
        action = sampled_batched["action"].to(self.device)

        # Forward pass
        action_out = self.model(state)

        # position velocity loss (dx, dy)
        pos_loss = self.loss(action_out[:, self.action_pos_idx], action[:, self.action_pos_idx])

        # angular velocity loss (dtheta)
        rot_loss = self.loss(action_out[:, self.action_rot_idx], action[:, self.action_rot_idx])

        total_loss = self.alpha_pos * pos_loss + self.alpha_rot * rot_loss

        # Log validation loss
        self.log(
            "val_pos_loss",
            pos_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "val_rot_loss",
            rot_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "val_loss",
            total_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

        return total_loss

    def on_after_backward(self):
        """Logs gradient norms after backpropagation but before optimizer step."""
        if self.debug:
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"⚠️ NaN gradient detected after backward: {name}")
        if self.trainer.global_step % 30 == 0:
            grad_norms = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    grad_norms += p.grad.data.norm(2).pow(2).item()

            wandb.log({"grad_norms": grad_norms}, step=self.global_step)

            weight_norms = 0.0
            for name, p in self.named_parameters():
                if "weight" in name:
                    weight_norms += p.data.norm(2).item()
            wandb.log({"weight_norms": weight_norms}, step=self.global_step)

    def on_validation_epoch_end(self):
        if self.current_epoch % self.evaluation_period == 0:
            self.create_model_quiver_plot()
            self.evaluate_model_in_env(debug=False)

    def on_fit_end(self):
        self.create_model_quiver_plot()
        self.evaluate_model_in_env(debug=False)

        ckpt_path = self.log_checkpoints()
        if self.record_final_video:
            self.evaluate_final_model_mushroom(ckpt_path)

    def create_model_quiver_plot(self):
        # Generate the quiver plot every 10 epochs
        save_path = self.plots_dir / f"quiver_plot_epoch_{self.current_epoch}.png"
        if self.env_id == "InsertionPandaLShapeRot":
            fig1, fig2 = generate_taskspace_quiver_plot(
                model=self.model, save_path=save_path, checkpoint_path=None, action_pos_idx=self.env.action_pos_idxs,
                action_rot_idx=self.env.action_rot_idxs, is_2d=self.env.is_2d, panda=True
            )
        else:
            fig1, fig2 = generate_taskspace_quiver_plot(
                model=self.model, save_path=save_path, checkpoint_path=None, action_pos_idx=self.env.action_pos_idxs,
                action_rot_idx=self.env.action_rot_idxs, is_2d=self.env.is_2d,
            )
        wandb.log({"quiver_plot": wandb.Image(fig1)})
        wandb.log({"stream_plot": wandb.Image(fig2)})
        plt.close(fig1)
        plt.close(fig2)

    def evaluate_model_in_env(self, debug):
        """
        Uses the current weights and runs trials in the named environment
        """
        if self.env is None or debug is not False:
            self.env: InsertionPegHole = Environment.make(self.env_id, debug_gui=debug)

        success_count = 0
        pos_distance = []
        rot_distance = []

        for i in range(self.num_trials):
            state = self.env.reset()[0]
            done = False
            curr_step = 0
            while not done and curr_step < self.max_env_steps:
                state_tensor = to_torch(state, device=self.device).unsqueeze(0)
                action = to_numpy(self.model(state_tensor)).squeeze()

                state, reward, done, info = self.env.step(action)
                curr_step = curr_step + 1

                if done or curr_step >= self.max_env_steps:
                    if done:
                        success_count += 1
                    else:
                        goal_position = self.env.peg_pos_goal
                        goal_rotation = self.env.peg_rot_goal.as_matrix()

                        final_position = state[:3]
                        final_rotation = state[3 : 3 + 9].reshape(3, 3)

                        pos_error = np.linalg.norm(
                            np.array(goal_position) - np.array(final_position)
                        )
                        pos_distance.append(pos_error)

                        rot_error_matrix = (
                            R.from_matrix(goal_rotation)
                            * R.from_matrix(final_rotation).inv()
                        )
                        rot_error_scalar = np.linalg.norm(rot_error_matrix.as_rotvec())
                        rot_distance.append(rot_error_scalar)

        # Create metrics for logging
        success_rate = success_count / self.num_trials
        pos_mean_distance = np.mean(pos_distance) if pos_distance else 0.0
        pos_std_distance = np.std(pos_distance) if pos_distance else 0.0

        rot_mean_distance = np.mean(rot_distance) if rot_distance else 0.0
        rot_std_distance = np.std(rot_distance) if rot_distance else 0.0

        # Logging
        wandb.log(
            {
                "epoch": self.current_epoch,
                "success_rate": success_rate,
                "pos_mean_distance_to_goal": pos_mean_distance,
                "pos_std_distance_to_goal": pos_std_distance,
                "rot_mean_distance_to_goal": rot_mean_distance,
                "rot_std_distance_to_goal": rot_std_distance,
            }
        )

    def log_checkpoints(self):
        # Save the latest model
        model_dir = self.logging_dir / "models"
        os.makedirs(model_dir, exist_ok=True)

        ckpt_path = Path(
            model_dir, self.checkpoint_filename_prefix + f"_{int(time.time())}.ckpt"
        )
        self.trainer.save_checkpoint(ckpt_path)

        # Create a WandB artifact
        artifact_name = os.path.basename(ckpt_path)
        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_file(str(ckpt_path))

        # Log checkpoint to WandB
        wandb.run.log_artifact(artifact)
        return ckpt_path

    def evaluate_final_model_mushroom(self, ckpt_path):
        videos_dir = (self.logging_dir / "vids")
        dataset, success_rate, _, _ = record_policy_behaviour(
            ckpt_path, self.env_id, num_episodes=self.num_rec_episodes,
            vid_dir=videos_dir,
            num_features=self.num_features,
            model_type=self.trainable_network,
        )

        # Extract the most recent directory from the videos_dir, as this hold the most recent video for logging
        os.makedirs(videos_dir.resolve(), exist_ok=True)
        video_dir = return_most_recent_file(videos_dir, True)

        # Create full path to video
        video_dir_path = Path(videos_dir, video_dir)
        video_files = [f for f in os.listdir(video_dir_path) if f.endswith(".mp4")]
        print(f"VideoFiles: {video_files}")

        if video_files:
            # Get the first mp4 file
            video_path = Path(video_dir_path, video_files[0])
            print(f"VidPath: {video_path}")
            # Replace part of the path
            output_video = Path(str(video_path).replace(".mp4", "_converted.mp4"))

            print(f"Output: {output_video}")
            ffmpeg.input(str(video_path)).output(
                str(output_video), vcodec="libx264", acodec="aac"
            ).run()
            print("Changed")

            # Only log if the mp4 file exists
            wandb.log({"final_video": wandb.Video(str(output_video), format="mp4")})
            wandb.log({"video_success_rate": success_rate})
