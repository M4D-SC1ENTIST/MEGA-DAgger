import os

import numpy as np

statesSeqMapping = {
    'poses_x': 'pose_x',
    'poses_y': 'pose_y',
    'poses_theta': 'pose_theta',
    'linear_vels_x': 'v'
}

# rollout_states = ('ego_x', 'ego_y', 'ego_theta', 'ego_v', 'opp_x', 'opp_y', 'opp_theta', 'opp_v',
#              'ego_best_traj_idx', 'opp_best_traj_idx', 'ego_weights', 'opp_weights',
#              'ego_prev_traj', 'ego_prev_opp', 'ego_ittc',
#              'objective')
rollout_states = ('rollout_obs', 'rollout_ego_s', 'rollout_opp_s',
                  'ego_best_traj_idx', 'opp_best_traj_idx',
                  'ego_prev_traj', 'ego_prev_opp', 'ego_ittc', 'ego_control_error')


def obsDict2carStateSeq(obs, ego_idx, state_keys=('poses_x', 'poses_y', 'poses_theta', 'linear_vels_x')):
    """
    Input:
    obs: [timestep_0, timestep_1, ...],
         inside each timestep:{'pose_x':[ego, opp_1, opp_2, ...], 'pose_y':[ego, opp_1, opp_2, ...]}

    Return:
    state for specific car: {'pose_x': [timestep_0, timestep_1]}
    """
    ego_states = {}
    for state in state_keys:
        state_seq_key = statesSeqMapping[state]
        ego_states[state_seq_key] = []
        for step in obs:
            ego_states[state_seq_key].append(np.round(step[state][ego_idx], 3))
    return ego_states


def paramsDict2Array(params, ord):
    res = []
    for key in ord:
        value = params[key]
        if type(value) != np.ndarray:
            res.append(params[key])
        else:
            for v in value:
                res.append(v)
    return np.array(res)


class RolloutDataLogger:
    def __init__(self, rollout_states=rollout_states, log_location=None):
        self.rollout_states = set(rollout_states)
        self.log_location = log_location
        self.rollout_buffer = {}
        for s in self.rollout_states:
            self.rollout_buffer[s] = []

    def getState(self, rollout_obs):
        ego_state = obsDict2carStateSeq(rollout_obs, 0)
        opp_state = obsDict2carStateSeq(rollout_obs, 1)
        mp = {
            'ego_x': ego_state['pose_x'],
            'ego_y': ego_state['pose_y'],
            'ego_theta': ego_state['pose_theta'],
            'ego_v': ego_state['v'],
            'opp_x': opp_state['pose_x'],
            'opp_y': opp_state['pose_y'],
            'opp_theta': opp_state['pose_theta'],
            'opp_v': opp_state['v']
        }
        return mp

    def clear_buffer(self):
        for s in self.rollout_states:
            self.rollout_buffer[s] = []

    def update_buffer(self, step_states):
        for s in step_states.keys():
            self.rollout_buffer[s].append(step_states[s])

    def remove_nan(self, key):
        tmp = self.rollout_buffer[key]
        tmp = np.array(tmp)
        tmp = tmp[~np.isnan(tmp)]
        self.rollout_buffer[key] = list(tmp)

    def save_rollout_data(self, rollout_obs=None, rollout_states=None, epoch=None, o_idx=None, s_idx=None, **kwargs):
        if not rollout_obs:
            rollout_obs = self.rollout_buffer['rollout_obs']
        if not rollout_states:
            rollout_states = self.rollout_buffer
            del rollout_states['rollout_obs']
        # log states
        mp = self.getState(rollout_obs)
        # log other
        mp.update(rollout_states)
        if kwargs:
            mp.update(kwargs)
        if epoch:
            npz_file = os.path.join(self.log_location, f'epoch_{epoch}', f's_{s_idx}_o_{o_idx}.npz')
        else:
            npz_file = os.path.join(self.log_location, f's_{s_idx}_o_{o_idx}.npz')
        np.savez_compressed(npz_file, **mp)





