# VideoMambaPro
Official Implementation of VideoMambaPro
![flowchart](fig/fig1.png)

# Installation

The required packages are in the file `requirements.txt`, and you can run the following command to install the environment

```
conda create --name videomae python=3.8 -y
conda activate videomambapro

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

pip install -r requirements.txt
```

### Note:
- **The above commands are for reference only**, please configure your own environment according to your needs.
- We recommend installing **`PyTorch >= 1.12.0`**, which may greatly reduce the GPU memory usage.
- It is recommended to install **`timm == 0.4.12`**, because some of the APIs we use are deprecated in the latest version of timm.
- We have supported pre-training with `PyTorch 2.0`, but it has not been fully tested.


# Data Preparation
We read and process the same way as [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/DATASET.md), but with a different convention for the format of the data list file. 


## Pre-train Dataset
The pretrain dataset loads the data list file, and then process each line in the list. The pre-training data list file is in the following format:

for video data line:
> video_path 0 -1

for rawframes data line:
> frame_folder_path start_index total_frames

For example, the UnlabeledHybrid data list file containing data from multiple sources, in part:
```
# The path prefix 'your_path' can be specified by `--data_root ${PATH_PREFIX}` in scripts when training or inferencing.

your_path/k400/---QUuC4vJs.mp4 0 -1
your_path/k400/--VnA3ztuZg.mp4 0 -1
...
your_path/AVA/frames/clip/zlVkeKC6Ha8 9601 300
your_path/AVA/frames/clip/zlVkeKC6Ha8 9901 300
...
your_path/SSv2/frames/182040 1 58
your_path/SSv2/frames/197728 1 29
...
```
where the AVA and Something-Something data are rawframes and the rest are videos.
