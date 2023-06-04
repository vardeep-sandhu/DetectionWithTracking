import os
import numpy as np
from AB3DMOT.AB3DMOT_libs.model import AB3DMOT
import argparse

def save_tracking_results_per_frame(filename, tracked_det):
    np.save(filename, tracked_det)


def perform_tracking(det_files, save_dir):
    # dataset is the txt data file of detections
    mot_tracker = AB3DMOT()
    np.set_printoptions(suppress=True)
    
    for idx, frame in enumerate(det_files):
        tracked_dets = []
        # dets are the detections in that frame of format [h, w, l, x, y, z, theta]
        dets = np.load(frame)
        dets[:, [5, 4, 3, 0, 1, 2, 6]] = dets[:, [0, 1, 2, 3, 4, 5, 6]]
    
        additional_info = np.zeros_like(dets)
        dets_all = {'dets': dets, 'info': additional_info}
        trackers = mot_tracker.update(dets_all)

        for d in trackers:
            bbox3d_tmp = d[0:7]
    
        # Saved tracked results below are in format:
        # [frame, id, obj_class, 7X[zeros], x, y, z, l, w, h, theta]
            one_track = [bbox3d_tmp[5], bbox3d_tmp[4], bbox3d_tmp[3],
                         bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2],  bbox3d_tmp[6]]
            tracked_dets.extend(one_track)

        tracked_det = np.array(tracked_dets).reshape(-1, 7)
        save_tracking_results_per_frame(f"{save_dir}/{(str(idx).zfill(4))}", tracked_det)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--save_path', '-s', type=str, default=None, required=True,
                        help='specify the path where you want to save the output')
    parser.add_argument('--det_path', '-d', type=str, default=None, required=True,
                        help='specify the detection directory')

    args = parser.parse_args()
    return args

    """
    The detections in .npy format are loaded.
    """
if __name__ == '__main__':
    args = parse_config()
    detection_path = args.det_path  
    detection_files = sorted([os.path.join(detection_path, file)
                              for file in os.listdir(detection_path)])
    
    assert os.path.isdir(detection_path), "The detection dir does not exist"
    assert len(detection_files) != 0, "The detection dir is empty."  
    
    save_root = os.path.basename(args.save_path)
    
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    
    elif len(os.listdir(save_root)) != 0:
        for file_ in os.listdir(save_root):
            print(f"removing file {os.path.join(save_root, file_)}")    
            os.remove(os.path.join(save_root, file_))

    perform_tracking(detection_files, save_root)
