#!/usr/bin/env python3
import os
import glob

import click
import numpy as np
import open3d as o3d
import pykitti
from tqdm import tqdm
import utils
import math 

def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


def yield_velo_scans(velo_files):
    """Generator to parse velodyne binary files into arrays."""
    for file in velo_files:
        yield load_velo_scan(file)


def vel2ply(points, T_mat):
    pcd = o3d.geometry.PointCloud()
    points_xyz = points[:, :3]
    points_xyz = convert_bb_frame(points_xyz, T_mat, "global")
    points_i = points[:, -1].reshape(-1, 1)
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.full_like(points_xyz,
                                                         points_i))
    return pcd

def convert_bb_frame(points, Trans_mat, global_or_local: str):
    
    """ 
    This trans_mat converts points from current to 0 frame i.e. local to global frame
    arg global_or_local asks if you want to convert the boxes from local to global frame
    or from global to local. 
    Here we are converting from local to global
    """

    if global_or_local == "global":
        pass
    elif global_or_local == "local":
        Trans_mat = np.linalg.inv(Trans_mat)

    if points.shape[1] == 7:
        xyz = utils.points_to_homo(points[: , :3])
        
        points_w = Trans_mat @ xyz.T

        points_w = np.delete(points_w.T, -1, axis=1)
        heading = np.copy(points[:, -1])

        R = (Trans_mat)[:3, :3]
        
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular :
            z = math.atan2(R[1,0], R[0,0])
        else :
            z = 0
        return np.hstack((points_w, points[:, 3:-1], (heading + z).reshape(-1,1)))
    
    if points.shape[1] == 3:
        
        xyz = utils.points_to_homo(points)
        points_w = Trans_mat @ xyz.T
        points[: , :3] = np.delete(points_w.T, -1, axis=1)
        return points


@click.command()
@click.option('--dataset',
              '-d',
              type=click.Path(exists=True),
              default=os.environ['HOME'] + '/data/kitti-odometry/dataset/',
              show_default=True,
              help='Location of the KITTI dataset')
@click.option('--out_dir',
              '-o',
              type=click.Path(exists=True),
              default=os.environ['HOME'] + '/data/kitti-odometry/ply/',
              show_default=True,
              help='Where to store the results')
@click.option('--sequence',
              '-s',
              type=str,
              default=None,
              required=False,
              help='Sequence number')
def main(dataset, out_dir, sequence):
    """Utility script to convert from the binary form found in the KITTI
    odometry dataset to .ply files. The intensity value for each measurement is
    encoded in the color channel of the output PointCloud.

    If a given sequence it's specified then it assumes you have a clean copy of
    the KITTI odometry benchmark, because it uses pykitti. If you only have a
    folder with just .bin files the script will most likely fail.

    If no sequence is specified then it blindly reads all the *.bin file in the
    specified dataset directory
    """
    print("Converting .bin scans into .ply fromat from:{orig} to:{dest}".
          format(orig=dataset, dest=out_dir))

    data_path = "/media/sandhu/SSD/dataset/kitti/odometry/dataset"
    # sequence = "08"
    gt_poses = utils.get_gt_poses(data_path, sequence)

    # Read all the *.bin scans from the dataset folder
    base_path = os.path.join(out_dir, '')
    velo_files = sorted(glob.glob(os.path.join(dataset, '*.bin')))
    
    scans = yield_velo_scans(velo_files)
    count = 0

    for points, scan_name in tqdm(zip(scans, velo_files),
                                  total=len(velo_files)):
        pcd = vel2ply(points, gt_poses[count])

        stem = os.path.splitext(scan_name.split('/')[-1])[0]
        filename = base_path + stem + '.ply'
        o3d.io.write_point_cloud(filename, pcd)

        count += 1

if __name__ == "__main__":
    main()
