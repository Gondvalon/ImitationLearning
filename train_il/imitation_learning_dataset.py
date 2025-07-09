import h5py
import numpy as np
import torch


class ImitationLearningDataset(torch.utils.data.Dataset):
    """Multimodal Manipulation dataset."""

    def __init__(self, filepaths_l):
        """
        Args:
            hdf5_file (handle): h5py handle of the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = filepaths_l
        self.dataset = {}

        # Store trajectory metadata
        self.states = []

        # Load metadata
        for file_path in self.dataset_path:
            with h5py.File(file_path, "r") as h5_file:
                for key in h5_file.keys():
                    if key.startswith("states_") and isinstance(
                        h5_file[key], h5py.Dataset
                    ):
                        # Collect trajectory information (file path, dataset key, and length)
                        length = h5_file[key].shape[0]

                        for idx in range(length):
                            self.states.append((file_path, key, idx))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        """
        Extract state action pair from dataset
        """
        file_path, key, state_idx = self.states[idx]

        trajectory_number = key.split("_")[-1]
        with h5py.File(file_path, "r") as h5_file:
            sa_pair = {
                "state": (h5_file[key][state_idx]).astype(np.float32),
                "action": (h5_file["actions_" + trajectory_number][state_idx]).astype(
                    np.float32
                ),
            }
        return sa_pair
