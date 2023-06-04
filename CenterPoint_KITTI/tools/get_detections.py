import argparse
from pathlib import Path
import open3d as o3d
import torch
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# from visual_utils import visualize_utils as V
from dataset import DemoDataset
from visual_utils.visualize_utils import boxes_to_corners_3d


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/kitti_models/pv_rcnn.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='data/custom/lidar_np',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default="checkpoints/pv_rcnn_8369.pth",
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy',
                        help='specify the extension of your point cloud data file')
    parser.add_argument('--save_path', type=str, required=True,
                        help='specify the save path of the generated detections')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def points2pcd(points, intensities=None):

    pcd = o3d.geometry.PointCloud()
    # colors = np.full_like(points, intensities.reshape(-1, 1))
    # print(points)
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


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
    lines = [(1, 0), (5, 4), (2, 3), (6, 7), (1, 2), (5, 6),
             (0, 3), (4, 7), (1, 5), (0, 4), (2, 6), (3, 7)]

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


def cloud_n_detections(boxes, color, points):
    points = points[:, :3]
    pcd = points2pcd(points)
    o3d_box = draw_3dbox(boxes, color)
    return [pcd] + o3d_box, points, boxes


def custom_vis(visual, window_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for i in visual:
        vis.add_geometry(i)
    # vis.get_render_option().load_from_json("render_o3d.json")
    vis.run()
    vis.destroy_window()


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info(
        '-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Processing sample: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            # print(pred_dicts)
            cloud = data_dict['points'][:, 1:].cpu().numpy()
            detections = pred_dicts[0]['pred_boxes'].cpu().numpy()
            confidence = pred_dicts[0]['pred_scores'].cpu().numpy()
            # detections = detections[confidence > 0.5]
            labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            if len(detections) > 0:
                np.save(f"{args.save_path}/{str(idx).zfill(4)}", detections)
                print("Saved as: ", str(idx).zfill(4))
            else:
                print("not saved")
            # print("Confidence ", confidence)
            # print("Labels ", labels)

            # visual, _, _ = cloud_n_detections(detections, "red", cloud)
            # custom_vis(visual, "")
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
