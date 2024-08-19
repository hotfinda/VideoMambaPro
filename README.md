# VideoMambaPro
Official Implementation of [VideoMambaPro: A Leap Forward for Mamba in Video Understanding](https://arxiv.org/abs/2406.19006)
![flowchart](fig/fig1.png)

we investigate similarities and differences of self-attention and Mamba from the perspective of the latter, and reveal the limitations of Mamba on video understanding task. We propose VideoMambaPro that uses [VideoMamba](https://github.com/OpenGVLab/VideoMamba) as a backbone, but significantly enhancing performance in the video understanding task, narrowing the gap with transformers. 
# Installation

The required packages are in the file `requirements.txt`, and you can run the following command to install the environment

```
conda create -n videomambapro python=3.10
conda activate videomambapro

conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging

pip install -r requirements.txt
```
pip install causal_conv1d==1.4.0 (we recommend to install through .whl file)
pip install mamba-ssm

# Data Preparation
We read and process the same way as [VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/DATASET.md), but with a different convention for the format of the data list file. 


## Pre-train Dataset
We pretrain the model on ImageNet-1K dataset, where the model loads a data list file with the following format:
> frame_folder_path total_frames label

## Fine-tune Dataset
There are two implementations of our finetune dataset `VideoClsDataset` and `RawFrameClsDataset`, supporting video data and rawframes data, respectively. Where SSV2 uses `RawFrameClsDataset` by default and the rest of the datasets use `VideoClsDataset`.

`VideoClsDataset` loads a data list file with the following format:
> video_path label

while `RawFrameClsDataset` loads a data list file with the following format:
> frame_folder_path total_frames label

For example, video data list and rawframes data list are shown below:
```
# The path prefix 'your_path' can be specified by `--data_root ${PATH_PREFIX}` in scripts when training or inferencing.

# k400 video data validation list
your_path/k400/jf7RDuUTrsQ.mp4 325
your_path/k400/JTlatknwOrY.mp4 233
your_path/k400/NUG7kwJ-614.mp4 103
your_path/k400/y9r115bgfNk.mp4 320
your_path/k400/ZnIDviwA8CE.mp4 244
...

# ssv2 rawframes data validation list
your_path/SomethingV2/frames/74225 62 140
your_path/SomethingV2/frames/116154 51 127
your_path/SomethingV2/frames/198186 47 173
your_path/SomethingV2/frames/137878 29 99
your_path/SomethingV2/frames/151151 31 166
...
```
# Codes details
Our project is based on VideoMamba for fair comparison. To solve limitation 1&2 in our paper, we mainly change the pipeline of Mamba by applying the diagonal mask during the backward SSM and applying residual connection on the bidirection SSM.
The  residual connection of Ab is realized through assign new matrix A in mamba/mamba_ssm/ops/selective_scan_interface.py
```
A = deltaA[:, :, i] + deltaA[:, :, x.index]
```
The mask assignment is realized through setting elements of A_b in mamba/mamba_ssm/modules/mamba_simple.py
```
self.A_b_log = mask_diagnomal (A_b_log)
```
