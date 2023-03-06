import numpy as np
from numpy.linalg import norm

from scipy import spatial

def cosine_similarity(A, B):
    return np.dot(A,B)/(norm(A, axis=1)*norm(B))

all_lidar_obs = np.load('models/lidar_data.npy')
print("all_lidar_obs shape: ", all_lidar_obs.shape)

snippet_lidar_obs = all_lidar_obs[0:100]

target_lidar_obs = all_lidar_obs[0]
test_noise = np.random.uniform(-0.25, 0.25, 108)
target_lidar_obs_with_noise = target_lidar_obs + test_noise

# test_cos_sim = cosine_similarity(snippet_lidar_obs, target_lidar_obs_with_noise)
cos_sim_arr = cosine_similarity(snippet_lidar_obs, target_lidar_obs)
#print("test_cos_sim: ", cos_sim_arr)

similarity_threshold=0.9993
indices_above_sim_thresh = np.where(cos_sim_arr > similarity_threshold)[0]
print("indices_above_sim_thresh: ", indices_above_sim_thresh)

"""
lidar_obs = np.ones((108))
print("lidar_obs shape: ", lidar_obs.shape)

test_noise = np.random.uniform(-0.25, 0.25, 108)
print("test_noise shape: ", test_noise.shape)
lidar_obs_with_noise = lidar_obs + test_noise
print("lidar_obs_with_noise shape: ", lidar_obs_with_noise.shape)

test_cos_sim = cosine_similarity(lidar_obs, lidar_obs_with_noise)
print("test_cos_sim: ", test_cos_sim)
"""