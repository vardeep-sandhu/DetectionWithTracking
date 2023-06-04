#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: some utilities

import os
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from visual_utils.visualize_utils import boxes_to_corners_3d
import open3d as o3d
np.random.seed(0)


def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
      Args:
        pose_path: (Complete) filename for the pose file
      Returns:
        A numpy array of size nx4x4 with n poses as 4x4 transformation
        matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']

    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')

    return np.array(poses)


def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)


def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds
      Returns:
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    # current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    # depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    proj_x_orig = np.copy(proj_x)

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    proj_y_orig = np.copy(proj_y)

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                         dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                          dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                       dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                             dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array(
        [scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity

    return proj_range, proj_x_orig, proj_y_orig, proj_idx


def gen_normal_map(current_range, current_vertex, proj_H=64, proj_W=900):
    """ Generate a normal image given the range projection of a point cloud.
      Args:
        current_range:  range projection of a point cloud, each pixel contains the corresponding depth
        current_vertex: range projection of a point cloud,
                        each pixel contains the corresponding point (x, y, z, 1)
      Returns:
        normal_data: each pixel contains the corresponding normal
    """
    normal_data = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)

    # iterate over all pixels in the range image
    for x in range(proj_W):
        for y in range(proj_H - 1):
            p = current_vertex[y, x][:3]
            depth = current_range[y, x]

            if depth > 0:
                wrap_x = wrap(x + 1, proj_W)
                u = current_vertex[y, wrap_x][:3]
                u_depth = current_range[y, wrap_x]
                if u_depth <= 0:
                    continue

                v = current_vertex[y + 1, x][:3]
                v_depth = current_range[y + 1, x]
                if v_depth <= 0:
                    continue

                u_norm = (u - p) / np.linalg.norm(u - p)
                v_norm = (v - p) / np.linalg.norm(v - p)

                w = np.cross(v_norm, u_norm)
                norm = np.linalg.norm(w)
                if norm > 0:
                    normal = w / norm
                    normal_data[y, x] = normal

    return normal_data


def wrap(x, dim):
    """ Wrap the boarder of the range image.
    """
    value = x
    if value >= dim:
        value = (value - dim)
    if value < 0:
        value = (value + dim)
    return value


def euler_angles_from_rotation_matrix(R):
    """ From the paper by Gregory G. Slabaugh, Computing Euler angles from a rotation matrix,
      psi, theta, phi = roll pitch yaw (x, y, z).
      Args:
        R: rotation matrix, a 3x3 numpy array
      Returns:
        a tuple with the 3 values psi, theta, phi in radians
    """

    def isclose(x, y, rtol=1.e-5, atol=1.e-8):
        return abs(x - y) <= atol + rtol * abs(y)

    phi = 0.0
    if isclose(R[2, 0], -1.0):
        theta = math.pi / 2.0
        psi = math.atan2(R[0, 1], R[0, 2])
    elif isclose(R[2, 0], 1.0):
        theta = -math.pi / 2.0
        psi = math.atan2(-R[0, 1], -R[0, 2])
    else:
        theta = -math.asin(R[2, 0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
        phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
    return psi, theta, phi


def load_vertex(scan_path):
    """ Load 3D points of a scan. The fileformat is the .bin format used in
      the KITTI dataset.
      Args:
        scan_path: the (full) filename of the scan file
      Returns:
        A nx4 numpy array of homogeneous points (x, y, z, 1).
    """
    current_points = np.load(scan_path).reshape((-1, 3))
    # current_vertex = np.empty((len(current_points), 4))

    # current_vertex = np.ones(
    #     (current_points.shape[0], current_points.shape[1] + 1))
    # current_vertex[:, :-1] = current_points
    return current_points


def load_radar_scenes(scan_path):
    """ Load Radar Scenes file which is in KITTI format
    as .bin. It has x_local, y_local, x_global, y_global

      Args:
        scan_path: the (full) filename of the scan file
      Returns:
        A nx4 numpy array of homogeneous points (x, y, 0, 1).
    """
    current_vertex = np.fromfile(scan_path, dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))
    xy_points = current_vertex[:, 0:2]
    z_zeros = np.zeros_like(len(current_vertex), dtype=np.float32)

    current_points = np.insert(xy_points, 2, z_zeros, axis=1)
    current_vertex = np.ones(
        (current_points.shape[0], current_points.shape[1] + 1))
    current_vertex[:, :-1] = current_points
    return current_vertex


def load_files(folder):
    """ Load all files in a folder and sort.
    """
    file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(folder)) for f in fn]
    file_paths.sort()
    return file_paths


def load_detections(det_path):
    det = np.load(det_path)
    return det


def load_labels(label_path):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half

    # sanity check
    assert ((sem_label + (inst_label << 16) == label).all())
    assert sem_label.max() in [0, 1] and sem_label.min() in [
        0, 1], "Something wrong with labels"

    return sem_label, inst_label


def draw_3dbox(bbox_coords, color):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    This is how the corners are retured 
    Input: NX7 Bounding box corrds
    Output: Open3d Lineset object
    """
    visuals = []

    points = boxes_to_corners_3d(bbox_coords)
    # lines = [(1, 0), (5, 4), (2, 3), (6, 7), (1, 2), (5, 6),
    #  (0, 3), (4, 7), (1, 5), (0, 4), (2, 6), (3, 7)]
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    if color == "red":
        colors = [[1, 0, 0] for i in range(len(lines))]
    if color == "something":
        colors = [[0, 1, 0] for i in range(len(lines))]
    for i in points:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(i)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        visuals.append(line_set)
    return visuals


def rotation_matrix_from_euler_angles(yaw, degrees=True):
    """ Generate rotation matrix given yaw angle.
      Args:
        yaw: yaw angle
      Returns:
        rotation matrix
    """
    return R.from_euler('z', yaw, degrees=degrees).as_matrix()


def gen_transformation(yaw, translation):
    """ Generate transformation from given yaw angle and translation.
      Args:
        current_range: range image
        current_vertex: point clouds
      Returns:
        normal image
    """
    rotation = R.from_euler('zyx', [[yaw, 0, 0]], degrees=True)
    rotation = rotation.as_dcm()[0]
    transformation = np.identity(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = [translation[0], translation[1], translation[2]]

    return transformation


def load_poses_kitti(pose_file, calib_file):
    """ load poses in kitti format """
    # laod poses
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # # convert poses in LiDAR coordinate system
    # new_poses = []
    # for pose in poses:
    #   new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    # new_poses = np.array(new_poses)
    # poses = new_poses

    return poses


def save_kitti_poses(out_path, poses):
    """ save poses in kitti format in a txt file """
    poses = poses.reshape((-1, 16))
    np.savetxt(os.path.join(out_path), poses[:, :12])


def make_predictions(sem_predictions, idx, seq):
    base_path = f"./predictions/sequence/{str(seq).zfill(4)}/predictions"
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    inst_pred = np.zeros_like(sem_predictions, dtype=np.uint32)
    predictions = sem_predictions + ((inst_pred.astype(np.uint32)) << 16)
    output_filename = os.path.join(base_path, f"{str(idx).zfill(5)}.label")
    print(output_filename)

    # Sanity Check
    assert predictions.max() in [0, 1] and predictions.min() in [
        0, 1], "More values in labels than 0 and 1"
    predictions.astype(np.uint32).tofile(output_filename)
