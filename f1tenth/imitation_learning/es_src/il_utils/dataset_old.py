import numpy as np
import pandas as pd
import il_utils.utils.downsampling as downsampling


class Dataset(object):
    def __init__(self):
        self.poses_x = None
        self.poses_y = None
        self.poses_theta = None
        self.scans = None
        self.actions = None

    def add(self, data):
        assert data["poses_x"].shape[0] == data["poses_y"].shape[0]
        assert data["poses_x"].shape[0] == data["poses_theta"].shape[0]
        assert data["poses_x"].shape[0] == data["scans"].shape[0]
        assert data["poses_x"].shape[0] == data["actions"].shape[0]

        if self.poses_x is None:
            self.poses_x = data["poses_x"]
            self.poses_y = data["poses_y"]
            self.poses_theta = data["poses_theta"]
            self.scans = data["scans"]
            self.actions = data["actions"]
        else:
            self.poses_x = np.concatenate([self.poses_x, data["poses_x"]])
            self.poses_y = np.concatenate([self.poses_y, data["poses_y"]])
            self.poses_theta = np.concatenate([self.poses_theta, data["poses_theta"]])
            self.scans = np.concatenate([self.scans, data["scans"]])
            self.actions = np.concatenate([self.actions, data["actions"]])

    def sample(self, batch_size):
        idx = np.random.permutation(self.scans.shape[0])[:batch_size]
        return {"scans": self.scans[idx], "actions": self.actions[idx]}

    def get_num_of_total_samples(self):
        return self.scans.shape[0]
    
    def random_truncate(self, max_len):
        idx = np.random.permutation(self.scans.shape[0])[:max_len]
        self.poses_x = self.poses_x[idx]
        self.poses_y = self.poses_y[idx]
        self.poses_theta = self.poses_theta[idx]
        self.scans = self.scans[idx]
        self.actions = self.actions[idx]
    
    def save_npz(self, path):
        np.savez(path, poses_x=self.poses_x, poses_y=self.poses_y, poses_theta=self.poses_theta, scans=self.scans, actions=self.actions)