import time
import click
import numpy as np

import sys
from utils import *
import open3d as o3d
import pynput.keyboard as keyboard
from open3d.ml.vis import Visualizer, LabelLUT


class vis_mos_results:
    def __init__(self, scan_files, detection_files):
        # init file paths
        self.scan_files = scan_files
        self.detection_files = detection_files

        self.current_points = load_vertex(scan_files[0])

        self.curr_detections = load_detections(detection_files[0])

        # init visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.current_points)
        self.pcd.paint_uniform_color([0.5, 0.5, 0.5])

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-45, -45, -5), max_bound=(45, 45, 5)
        )
        self.pcd = self.pcd.crop(bbox)  # set view area

        self.vis.add_geometry(self.pcd)

        # init keyboard controller
        key_listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        key_listener.start()

        # init frame index
        self.frame_idx = 0
        self.num_frames = len(self.scan_files)

    def on_press(self, key):
        try:
            if key.char == "q":
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)

            if key.char == "n":
                if self.frame_idx < self.num_frames - 1:
                    self.frame_idx += 1
                    self.current_points = load_vertex(
                        self.scan_files[self.frame_idx])
                    self.curr_detections = load_detections(
                        self.detection_files[self.frame_idx])
                    print("frame index:", self.frame_idx)
                else:
                    print("Reach the end of this sequence!")

            if key.char == "b":
                if self.frame_idx > 1:
                    self.frame_idx -= 1
                    self.current_points = load_vertex(
                        self.scan_files[self.frame_idx])
                    self.curr_detections = load_detections(
                        self.detection_files[self.frame_idx])
                    print("frame index:", self.frame_idx)
                else:
                    print("At the start at this sequence!")

        except AttributeError:
            print("special key {0} pressed".format(key))

    def on_release(self, key):
        try:
            if key.char == "n" or key.char == "b":
                self.current_points = load_vertex(
                    self.scan_files[self.frame_idx])
                self.curr_detections = load_detections(
                    self.detection_files[self.frame_idx])

        except AttributeError:
            print("special key {0} pressed".format(key))

    def run(self):
        current_points = self.current_points
        current_detections = self.curr_detections

        self.pcd.points = o3d.utility.Vector3dVector(current_points)
        o3d_box = draw_3dbox(current_detections, "red")
        self.pcd.paint_uniform_color([0.5, 0.5, 0.5])

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-75, -75, -5), max_bound=(75, 75, 5)
        )

        self.pcd = self.pcd.crop(bbox)  # set view area

        self.vis.clear_geometries()
        geometry = [self.pcd] + o3d_box

        for i in geometry:
            self.vis.add_geometry(i, reset_bounding_box=False)

        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.1)


@click.command()
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    default="./data/custom/lidar_np",
)
@click.option(
    "--detections",
    "-det",
    type=click.Path(exists=True),
    default="./detections/",
)
# @click.option("--seq", "-seq", type=int, required=True, help="Add seq number that you wanna see. Make sure you have predictions for this seq")
def main(data, detections):
    scan_paths = load_files(data)
    detections_paths = load_files(detections)
    assert os.path.isdir(detections) or (
        len(detections_paths) == 0), "Told you make predictions first for this seq"
    visualizer = vis_mos_results(scan_paths, detections_paths)

    while True:
        visualizer.run()


if __name__ == "__main__":
    main()
