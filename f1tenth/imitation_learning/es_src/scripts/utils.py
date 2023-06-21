import math
import numpy as np
from pyglet.gl import GL_LINES

from f110_gym.envs.laser_models import trace_ray
from f110_gym.envs.collision_models import collision, get_vertices

def check_ttc(ego_scan, linear_vels_x, ttc_threshold=0.5, scan_num=1080):
    """
    Checks the time to collision (TTC) of the agent based on the lidar scan.
    ego

    Input: 
        ego_scan: Lidar scan of the ego vehicle (e.g.: obs['scan'][0])
        linear_vels_x: Linear velocity of the ego vehicle (e.g.: obs['linear_vels_x'][0])
        ttc_threshold: Threshold for detecting Time to Collision (TTC). 
                       The time is set to 0.0 beyond the detecting threshold.
                       Can be used to determine if the ego vehicle is close to collision.
        scan_num: Dimension of lidar scan
    Output:
        within_threshold: Whether the TTC is within the threshold (close to collision)
        abs_ittc: Time to collision
    """

    angle_span = np.linspace(-0.75 * np.pi, 0.75 * np.pi, scan_num)
    ego_speed_proj = np.cos(angle_span) * linear_vels_x
    ego_speed_proj[ego_speed_proj <= 0.0] = 0.001
    raw_ittc = ego_scan / ego_speed_proj
    if np.min(raw_ittc) > ttc_threshold:
        within_threshold = False
        abs_ittc = 0.0
    else:
        within_threshold = True
        abs_ittc = np.min(raw_ittc)

    return within_threshold, abs_ittc


def render_line(env, pt1_x, pt1_y, pt2_x, pt2_y, color=(255, 255, 0)):
    """
    Renders the line.
    Input:
        env: environment
        pt1_x: x coordinate of the first point
        pt1_y: y coordinate of the first point
        pt2_x: x coordinate of the second point
        pt2_y: y coordinate of the second point
        color: color of the separation line
    Output:
        vlist: vertex list
    """
    vlist = env.renderer.batch.add(2, GL_LINES, None, ('v2f', (pt1_x * 50., pt1_y * 50., pt2_x * 50., pt2_y * 50.)), ('c3B', color * 2))
    return vlist

def get_point_projection(x, y, theta, projected_dist=100):
    """
    Projects the point for separation line of the agent.
    Input:
        x: x coordinate of the agent
        y: y coordinate of the agent
        theta: orientation of the agent
        projected_dist: distance to project the separation line
    Output:
        pt2_x: x coordinate of the second point
        pt2_y: y coordinate of the second point
    """
    pt2_x = x + projected_dist * np.cos(theta)
    pt2_y = y + projected_dist * np.sin(theta)

    return pt2_x, pt2_y


def determine_direction(sep_pt1_x, sep_pt1_y, sep_pt2_x, sep_pt2_y, scan_pt_x, scan_pt_y):
    """
    Determines the direction of the agent based on the lidar scan.
    Input:
        sep_pt1_x: x coordinate of the first point of the separation line
        sep_pt1_y: y coordinate of the first point of the separation line
        sep_pt2_x: x coordinate of the second point of the separation line
        sep_pt2_y: y coordinate of the second point of the separation line
        scan_pt_x: x coordinate of the lidar scan point
        scan_pt_y: y coordinate of the lidar scan point
    Output:
        the current lidar point is left, right, or colinear with respect to the ego vehicle
    """
    # https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
    val = (sep_pt2_x - sep_pt1_x) * (scan_pt_y - sep_pt1_y) - (sep_pt2_y - sep_pt1_y) * (scan_pt_x - sep_pt1_x)
    if val > 0:
        return 'left'
    elif val < 0:
        return 'right'
    else:
        return 'colinear'


def get_theta_idx_from_rad(theta_rad, angle_increment):
    """
    Gets the index of the lidar scan from the angle.
    Input:
        theta_rad: angle of the lidar scan
        angle_increment: angle increment of the lidar scan
    Output:
        theta_idx: index of the lidar scan
    """
    if theta_rad > 2*np.pi:
        theta_rad = theta_rad - 2*np.pi
    elif theta_rad < 0:
        theta_rad = theta_rad + 2*np.pi
    theta_idx = int(theta_rad/angle_increment)

    return theta_idx

def get_rad_from_theta_idx(theta_idx, angle_increment):
    theta_rad = theta_idx * angle_increment
    if theta_rad > np.pi:
        theta_rad = theta_rad - 2 * np.pi

    return theta_rad

def get_perpendicular_boundary_coordinates(env, ego_x, ego_y, ego_theta):
    """
    Gets the boundary coordinates of the agent based on the lidar scan.
    Input:
        env: environment
        ego_x: x coordinate of the agent
        ego_y: y coordinate of the agent
        ego_theta: orientation of the agent
    Output:
        left_boundary_x: x coordinate of the left boundary
        left_boundary_y: y coordinate of the left boundary
        right_boundary_x: x coordinate of the right boundary
        right_boundary_y: y coordinate of the right boundary
        left_boundary_dist: distance of the left boundary
        right_boundary_dist: distance of the right boundary
    """
    # Get ego scan simulator
    ego_scan_sim = env.sim.agents[0].scan_simulator

    # Extract parameters from ego scan simulator
    theta_dis = ego_scan_sim.theta_dis
    fov = ego_scan_sim.fov
    sines = ego_scan_sim.sines
    cosines = ego_scan_sim.cosines
    eps = ego_scan_sim.eps
    orig_x = ego_scan_sim.orig_x
    orig_y = ego_scan_sim.orig_y
    orig_c = ego_scan_sim.orig_c
    orig_s = ego_scan_sim.orig_s
    height = ego_scan_sim.map_height
    width = ego_scan_sim.map_width
    resolution = ego_scan_sim.map_resolution
    dt = ego_scan_sim.dt
    max_range = ego_scan_sim.max_range
    num_beams = ego_scan_sim.num_beams

    angle_increment = fov / num_beams
 
    
    # Prepare angles
    left_offset_theta = np.pi / 2
    right_offset_theta = -np.pi / 2

    left_ray_theta = ego_theta + left_offset_theta
    right_ray_theta = ego_theta + right_offset_theta

    left_ray_theta = left_ray_theta % (2 * np.pi)
    right_ray_theta = right_ray_theta % (2 * np.pi)


    # Calculate necessary inputs for trace_ray from scan sim params
    # left_theta_index = theta_dis * (left_ray_theta - fov/2.)/(2. * np.pi)
    # right_theta_index = theta_dis * (right_ray_theta - fov/2.)/(2. * np.pi)
    # left_theta_index = get_theta_idx_from_rad(left_ray_theta, theta_dis, fov)
    # right_theta_index = get_theta_idx_from_rad(right_ray_theta, theta_dis, fov)
   
    left_theta_index = get_theta_idx_from_rad(left_ray_theta, angle_increment)
    right_theta_index = get_theta_idx_from_rad(right_ray_theta, angle_increment)

    print("left_theta_index: ", left_theta_index)
    print("right_theta_index: ", right_theta_index)

    # left_theta_index = np.fmod(left_theta_index, theta_dis)
    # right_theta_index = np.fmod(right_theta_index, theta_dis)

    # while (left_theta_index < 0):
    #     left_theta_index += theta_dis

    # while (right_theta_index < 0):
    #     right_theta_index += theta_dis

    

    left_boundary_dist = trace_ray(ego_x, ego_y, left_theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range)
    right_boundary_dist = trace_ray(ego_x, ego_y, right_theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range)
    

    left_boundary_x, left_boundary_y = get_point_projection(ego_x, ego_y, left_ray_theta, left_boundary_dist)
    right_boundary_x, right_boundary_y = get_point_projection(ego_x, ego_y, right_ray_theta, right_boundary_dist)
    return left_boundary_x, left_boundary_y, right_boundary_x, right_boundary_y, left_boundary_dist, right_boundary_dist


def get_closest_boundary_pts(env, ego_scan, ego_x, ego_y, ego_theta):
    """
    Gets the closest boundary points of the agent based on the lidar scan.
    Input:
        env: environment
        ego_scan: lidar scan of the agent
        ego_x: x coordinate of the agent
        ego_y: y coordinate of the agent
        ego_theta: orientation of the agent
    Output:
        left_pt_x: x coordinate of the left boundary
        left_pt_y: y coordinate of the left boundary
        right_pt_x: x coordinate of the right boundary
        right_pt_y: y coordinate of the right boundary
        closest_left_dist: distance to the left boundary
        closest_right_dist: distance to the right boundary
    """

    ego_scan_sim = env.sim.agents[0].scan_simulator

    num_beams = ego_scan_sim.num_beams
    fov = ego_scan_sim.fov

    angle_increment = fov / num_beams

    scan_len = len(ego_scan)
    mid_elem = int(scan_len / 2)

    left_scan_pts = ego_scan[:mid_elem]
    right_scan_pts = ego_scan[mid_elem:]

    closest_left_dist = min(left_scan_pts)
    closest_right_dist = min(right_scan_pts)
    
    closest_left_idx = np.argmin(left_scan_pts)
    closest_right_idx = np.argmin(right_scan_pts) + mid_elem

    closest_left_offset_theta = (closest_left_idx - mid_elem) * angle_increment
    closest_right_offset_theta = (closest_right_idx - mid_elem) * angle_increment

    # print("closest_left_offset_theta: ", closest_left_offset_theta)
    # print("closest_right_offset_theta: ", closest_right_offset_theta)
    
    closest_right_pt_rad = ego_theta + closest_left_offset_theta
    closest_left_pt_rad = ego_theta + closest_right_offset_theta

    right_pt_x, right_pt_y = get_point_projection(ego_x, ego_y, closest_right_pt_rad, closest_left_dist)
    left_pt_x, left_pt_y = get_point_projection(ego_x, ego_y, closest_left_pt_rad, closest_right_dist)

    return left_pt_x, left_pt_y, right_pt_x, right_pt_y, closest_left_dist, closest_right_dist


def get_front_closest_pt(ego_scan, ego_x, ego_y, ego_theta):
    front_closest_dist = ego_scan[int(len(ego_scan) / 2)]
    front_closest_pt_x, front_closest_pt_y = get_point_projection(ego_x, ego_y, ego_theta, front_closest_dist)
    return front_closest_pt_x, front_closest_pt_y, front_closest_dist


def check_target_pt_hit_oppo(env, target_pt_x, target_pt_y):
    # Extract information from simulator and opponent object
    sim = env.sim
    car_length = sim.params['length']
    car_width = sim.params['width']

    oppo = sim.agents[1]
    oppo_pose = np.append(oppo.state[0:2], oppo.state[4])

    # Get vertices of opponent agent and target point
    oppo_vert = get_vertices(oppo_pose, car_length, car_width)
    target_pt_vert = np.asarray([[target_pt_x, target_pt_y]])

    # Check collision
    collision_res = collision(oppo_vert, target_pt_vert)
    return collision_res


def get_vehicle_upper_right_coordinates(vehicle_x, vehicle_y, vehicle_theta, vehicle_length, vehicle_width):
    """
    Get the coordinates of the upper right corner of the vehicle
    :param vehicle_x: x coordinate of the vehicle
    :param vehicle_y: y coordinate of the vehicle
    :param vehicle_theta: angle of the vehicle
    :param vehicle_length: length of the vehicle
    :param vehicle_width: width of the vehicle
    :return: x and y coordinates of the upper right corner of the vehicle
    """
    upper_right_x = vehicle_x + (vehicle_length / 2) * np.cos(vehicle_theta) + (vehicle_width / 2) * np.sin(vehicle_theta)
    upper_right_y = vehicle_y + (vehicle_length / 2) * np.sin(vehicle_theta) - (vehicle_width / 2) * np.cos(vehicle_theta)
    return upper_right_x, upper_right_y


def get_vehicle_upper_left_coordinates(vehicle_x, vehicle_y, vehicle_theta, vehicle_length, vehicle_width):
    """
    Get the coordinates of the upper left corner of the vehicle
    :param vehicle_x: x coordinate of the vehicle
    :param vehicle_y: y coordinate of the vehicle
    :param vehicle_theta: angle of the vehicle
    :param vehicle_length: length of the vehicle
    :param vehicle_width: width of the vehicle
    :return: x and y coordinates of the upper left corner of the vehicle
    """
    upper_left_x = vehicle_x + (vehicle_length / 2) * np.cos(vehicle_theta) - (vehicle_width / 2) * np.sin(vehicle_theta)
    upper_left_y = vehicle_y + (vehicle_length / 2) * np.sin(vehicle_theta) + (vehicle_width / 2) * np.cos(vehicle_theta)
    return upper_left_x, upper_left_y


def get_vehicle_lower_right_coordinates(vehicle_x, vehicle_y, vehicle_theta, vehicle_length, vehicle_width):
    """
    Get the coordinates of the lower right corner of the vehicle
    :param vehicle_x: x coordinate of the vehicle
    :param vehicle_y: y coordinate of the vehicle
    :param vehicle_theta: angle of the vehicle
    :param vehicle_length: length of the vehicle
    :param vehicle_width: width of the vehicle
    :return: x and y coordinates of the lower right corner of the vehicle
    """
    lower_right_x = vehicle_x - (vehicle_length / 2) * np.cos(vehicle_theta) + (vehicle_width / 2) * np.sin(vehicle_theta)
    lower_right_y = vehicle_y - (vehicle_length / 2) * np.sin(vehicle_theta) - (vehicle_width / 2) * np.cos(vehicle_theta)
    return lower_right_x, lower_right_y


def get_vehicle_lower_left_coordinates(vehicle_x, vehicle_y, vehicle_theta, vehicle_length, vehicle_width):
    """
    Get the coordinates of the lower left corner of the vehicle
    :param vehicle_x: x coordinate of the vehicle
    :param vehicle_y: y coordinate of the vehicle
    :param vehicle_theta: angle of the vehicle
    :param vehicle_length: length of the vehicle
    :param vehicle_width: width of the vehicle
    :return: x and y coordinates of the lower left corner of the vehicle
    """
    lower_left_x = vehicle_x - (vehicle_length / 2) * np.cos(vehicle_theta) - (vehicle_width / 2) * np.sin(vehicle_theta)
    lower_left_y = vehicle_y - (vehicle_length / 2) * np.sin(vehicle_theta) + (vehicle_width / 2) * np.cos(vehicle_theta)
    return lower_left_x, lower_left_y


# def prepare_ego_oppo_coordinates_for_overtaking_check(env, ego_x, ego_y, ego_theta, oppo_x, oppo_y, oppo_theta):
#     sim = env.sim
#     car_length = sim.params['length']
#     car_width = sim.params['width']
# 
#     oppo_upper_right_x, oppo_upper_right_y = get_vehicle_upper_right_coordinates(oppo_x, oppo_y, oppo_theta, car_length, car_width)

#     return oppo_upper_right_x, oppo_upper_right_y

"""
def check_if_successful_overtake(env, ego_x, ego_y, ego_theta, oppo_x, oppo_y, oppo_theta, dist_thresh=0):
    sim = env.sim
    car_length = sim.params['length']
    car_width = sim.params['width']

    # Get ego and opponent corner coordinates
    ego_upper_left_x, ego_upper_left_y = get_vehicle_upper_left_coordinates(ego_x, ego_y, ego_theta, car_length, car_width)
    ego_upper_right_x, ego_upper_right_y = get_vehicle_upper_right_coordinates(ego_x, ego_y, ego_theta, car_length, car_width)

    ego_lower_left_x, ego_lower_left_y = get_vehicle_lower_left_coordinates(ego_x, ego_y, ego_theta, car_length, car_width)
    ego_lower_right_x, ego_lower_right_y = get_vehicle_lower_right_coordinates(ego_x, ego_y, ego_theta, car_length, car_width)

    oppo_upper_left_x, oppo_upper_left_y = get_vehicle_upper_left_coordinates(oppo_x, oppo_y, oppo_theta, car_length, car_width)
    oppo_upper_right_x, oppo_upper_right_y = get_vehicle_upper_right_coordinates(oppo_x, oppo_y, oppo_theta, car_length, car_width)

    oppo_lower_left_x, oppo_lower_left_y = get_vehicle_lower_left_coordinates(oppo_x, oppo_y, oppo_theta, car_length, car_width)
    oppo_lower_right_x, oppo_lower_right_y = get_vehicle_lower_right_coordinates(oppo_x, oppo_y, oppo_theta, car_length, car_width)

    # Calculate distance between corners
    x_ego_lower_left_to_oppo_upper_right = ego_lower_left_x - oppo_upper_right_x
    y_ego_lower_left_to_oppo_upper_right = ego_lower_left_y - oppo_upper_right_y

    x_ego_lower_right_to_oppo_upper_left = ego_lower_right_x - oppo_upper_left_x
    y_ego_lower_right_to_oppo_upper_left = ego_lower_right_y - oppo_upper_left_y

    hypotenuse_ego_lower_left_to_oppo_upper_right = np.sqrt(x_ego_lower_left_to_oppo_upper_right ** 2 + y_ego_lower_left_to_oppo_upper_right ** 2)
    hypotenuse_ego_lower_right_to_oppo_upper_left = np.sqrt(x_ego_lower_right_to_oppo_upper_left ** 2 + y_ego_lower_right_to_oppo_upper_left ** 2)

    sine_ego_lower_left_to_oppo_upper_right = y_ego_lower_left_to_oppo_upper_right / hypotenuse_ego_lower_left_to_oppo_upper_right
    sine_ego_lower_right_to_oppo_upper_left = y_ego_lower_right_to_oppo_upper_left / hypotenuse_ego_lower_right_to_oppo_upper_left

    print("sine_ego_lower_left_to_oppo_upper_right: ", sine_ego_lower_left_to_oppo_upper_right)
    print("sine_ego_lower_right_to_oppo_upper_left: ", sine_ego_lower_right_to_oppo_upper_left)
"""

def check_if_overtake(env, oppo_scan, oppo_x, oppo_y, oppo_theta, lidar_range=2.0):
    # Get ego scan simulator
    ego_scan_sim = env.sim.agents[0].scan_simulator

    theta_dis = ego_scan_sim.theta_dis
    fov = ego_scan_sim.fov
    sines = ego_scan_sim.sines
    cosines = ego_scan_sim.cosines
    eps = ego_scan_sim.eps
    orig_x = ego_scan_sim.orig_x
    orig_y = ego_scan_sim.orig_y
    orig_c = ego_scan_sim.orig_c
    orig_s = ego_scan_sim.orig_s
    height = ego_scan_sim.map_height
    width = ego_scan_sim.map_width
    resolution = ego_scan_sim.map_resolution
    dt = ego_scan_sim.dt
    max_range = ego_scan_sim.max_range
    num_beams = ego_scan_sim.num_beams

    angle_increment = fov / num_beams

    sim = env.sim
    car_length = sim.params['length']
    car_width = sim.params['width']

    ego = sim.agents[0]
    ego_pose = np.append(ego.state[0:2], ego.state[4])

    ego_vert = get_vertices(ego_pose, car_length, car_width)
    
    """
    # Prepare angles
    left_offset_theta = lidar_range / 2
    right_offset_theta = -lidar_range / 2

    left_ray_theta = oppo_theta + left_offset_theta
    right_ray_theta = oppo_theta + right_offset_theta

    left_ray_theta = left_ray_theta % (2 * np.pi)
    right_ray_theta = right_ray_theta % (2 * np.pi)
   
    left_theta_index = get_theta_idx_from_rad(left_ray_theta, angle_increment)
    right_theta_index = get_theta_idx_from_rad(right_ray_theta, angle_increment)

    
    if left_theta_index > right_theta_index:
        extracted_lidar_scan = oppo_scan[right_theta_index:left_theta_index]
    else:
        extracted_lidar_scan = oppo_scan[left_theta_index:right_theta_index]

    angle_increment_idx = 0
    curr_theta = left_ray_theta

    scan_x_projection = []
    scan_y_projection = []
    # scan_dist = []

    for i in range(extracted_lidar_scan.shape[0]):
        curr_scan_dist = trace_ray(oppo_x, oppo_y, left_theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range)

        curr_x = oppo_x + curr_scan_dist * np.cos(curr_theta)
        curr_y = oppo_y + curr_scan_dist * np.sin(curr_theta)

        

        scan_x_projection.append(curr_x)
        scan_y_projection.append(curr_y)
        # scan_dist.append(curr_scan_dist)
        
        angle_increment_idx += 1

        curr_lidar_pt_vert = np.asarray([[curr_x, curr_y]])

        collision_res = collision(ego_vert, curr_lidar_pt_vert)

        if collision_res:
            return True, scan_x_projection, scan_y_projection
    

    left_oppo_scan_dist = trace_ray(oppo_x, oppo_y, left_theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range)
    right_oppo_scan_dist = trace_ray(oppo_x, oppo_y, right_theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range)
    
    left_oppo_scan_x, left_oppo_scan_y = get_point_projection(oppo_x, oppo_y, left_ray_theta, left_oppo_scan_dist)
    right_oppo_scan_x, right_oppo_scan_y = get_point_projection(oppo_x, oppo_y, right_ray_theta, right_oppo_scan_dist)
    """
    scan_len = len(oppo_scan)
    mid_elem = int(scan_len / 2)

    offset_elem = int((fov - lidar_range) / angle_increment)
    offset_elem = int(offset_elem / 2)

    left_scan_pts = oppo_scan[offset_elem:mid_elem]
    right_scan_pts = oppo_scan[mid_elem:scan_len - offset_elem]

    closest_left_dist = min(left_scan_pts)
    closest_right_dist = min(right_scan_pts)
    
    closest_left_idx = np.argmin(left_scan_pts) + offset_elem
    closest_right_idx = np.argmin(right_scan_pts) + mid_elem

    closest_left_offset_theta = (closest_left_idx - mid_elem) * angle_increment
    closest_right_offset_theta = (closest_right_idx - mid_elem) * angle_increment

    closest_right_pt_rad = oppo_theta + closest_left_offset_theta
    closest_left_pt_rad = oppo_theta + closest_right_offset_theta

    right_pt_x, right_pt_y = get_point_projection(oppo_x, oppo_y, closest_right_pt_rad, closest_left_dist)
    left_pt_x, left_pt_y = get_point_projection(oppo_x, oppo_y, closest_left_pt_rad, closest_right_dist)

    left_target_pt = np.asarray([[left_pt_x, left_pt_y]])
    right_target_pt = np.asarray([[right_pt_x, right_pt_y]])

    left_collision_res = collision(ego_vert, left_target_pt)
    right_collision_res = collision(ego_vert, right_target_pt)

    # print("left_collision_res: ", left_collision_res)
    # print("right_collision_res: ", right_collision_res)

    if left_collision_res or right_collision_res:
        return True, left_pt_x, left_pt_y, right_pt_x, right_pt_y
    else:
        return False, left_pt_x, left_pt_y, right_pt_x, right_pt_y


def clip_res(collision_rate, overtake_rate):
    """
    If in some scenarios, the ego vehicle successful overtake while also collide, count it as overtake successful.
    (rarely happens, only seen when undesired_overtake_behavior_prob is 0)
    """
    rate_sum = collision_rate + overtake_rate
    if rate_sum > 1:
        gap = rate_sum - 1
        collision_rate -= gap
    
    return round(collision_rate, 2), round(overtake_rate, 2)