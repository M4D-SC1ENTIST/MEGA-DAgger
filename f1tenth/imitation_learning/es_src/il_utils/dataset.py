import numpy as np
import il_utils.utils.downsampling as downsampling


class Dataset(object):
    def __init__(self):
        self.poses_x = None
        self.poses_y = None
        self.poses_theta = None
        self.scans = None
        self.actions = None
        self.cbf_val = None
        self.vel_x = None
        self.vel_y = None

    def add(self, data):
        assert data["poses_x"].shape[0] == data["poses_y"].shape[0]
        assert data["poses_x"].shape[0] == data["poses_theta"].shape[0]
        assert data["poses_x"].shape[0] == data["scans"].shape[0]
        assert data["poses_x"].shape[0] == data["actions"].shape[0]
        assert data["poses_x"].shape[0] == data["cbf_val"].shape[0]
        assert data["poses_x"].shape[0] == data["vel_x"].shape[0]
        assert data["poses_x"].shape[0] == data["vel_y"].shape[0]


        if self.poses_x is None:
            self.poses_x = data["poses_x"]
            self.poses_y = data["poses_y"]
            self.poses_theta = data["poses_theta"]
            self.scans = data["scans"]
            self.actions = data["actions"]
            self.cbf_val = data["cbf_val"]
            self.vel_x = data["vel_x"]
            self.vel_y = data["vel_y"]
        else:
            self.poses_x = np.concatenate([self.poses_x, data["poses_x"]])
            self.poses_y = np.concatenate([self.poses_y, data["poses_y"]])
            self.poses_theta = np.concatenate([self.poses_theta, data["poses_theta"]])
            self.scans = np.concatenate([self.scans, data["scans"]])
            self.actions = np.concatenate([self.actions, data["actions"]])
            self.cbf_val = np.concatenate([self.cbf_val, data["cbf_val"]])
            self.vel_x = np.concatenate([self.vel_x, data["vel_x"]])
            self.vel_y = np.concatenate([self.vel_y, data["vel_y"]])

    def sample(self, batch_size):
        idx = np.random.permutation(self.scans.shape[0])[:batch_size]
        return {"scans": self.scans[idx], "actions": self.actions[idx]}

    def get_num_of_total_samples(self):
        return self.scans.shape[0]
