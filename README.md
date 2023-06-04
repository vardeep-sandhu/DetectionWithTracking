## Get started:

Its good to have a conda env for this task. You can make it like this:
```
conda create -n obj_det python=3.8
```
Then clone this repo by running the following command. 

```
git clone git@github.com:vardeep-motor-ai/DetectionWTracking.git
```

The dependecies can be installed by running:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
and 
```
pip install -r requirements.txt
```
Additionally 2 other packages need to be installed namely `spconv` and `pcdet`. We will install these one by one.

1. spconv:
With this version of centerpoint spconv v1.2 works. 

Install it as a python package inside CenterPoint_KITTI using the following commands:
```
git clone git@github.com:traveller59/spconv.git -b v1.2.1 --recursive
cd spconv/
python setup.py bdist_wheel
cd dist/
pip install spconv-1.2.1-cp38-cp38-linux_x86_64.whl
```

2. pcdet
This is quite simple. 

```
cd project_main_directory/CenterPoint_KITTI
python setup.py develop
```

This installs pcdet package. 

## Running the detection and evaluation pipeline. 

First, here we assume that we already have access to the .npy files of the individual merged lidar point clouds. These files reside in the `.data/` directory.

Then, we run the detection pipeline to get all the detections from these merged point clouds. We do this by running 

```
MODEL="pv_rcnn_8369"

rm -rf detections/$MODEL 
mkdir detections/$MODEL
python tools/get_detections.py --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml --data_path ../data/custom/lidar_np --ckpt ../checkpoints/$MODEL.pth --save_path detections/$MODEL
```
or alternativly by running `./detections_generate.sh`. Here `--cfg_file` indicates the model config file, `--data_path` is the path where the extracted lidar point cloud is placed, `-ckpt` is the path of the checkpoint and `--save_path` is the path where the detections will be saved.

## Tracking pipeline.

To do tracking on top of the detection pipeline, we follow the given steps. We do tracking to remove the false positive bounding boxes and to fill in the missed detections. We use `AB3DMOT` as our tracker. 

To get the tracking resulst we do:
```
python scripts/get_tracking.py -d CenterPoint_KITTI/detections/pv_rcnn_8369 -s tracking_results
```
where `-d` is the path to detections dir and `-s` is the save path of the tracking results. 

In the end you can visualize both tracking and detection files by running:

```
python tools/vis.py --data ../data/custom/lidar_np --detections detections/pv_rcnn_8369

```
or alternatively `./visualize_detections.sh`. Where, `--data` is the point cloud `.npy` files and `--detections` are the detection files (tracking files can also be visualized) the same way.

## Future Work

1. Integrating rosbag to numpy converter.
2. Making docker container 
3. Making workflow stramlined 