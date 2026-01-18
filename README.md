# MSSF: A 4D Radar and Camera Fusion Framework With Multi-Stage Sampling for 3D Object Detection in Autonomous Driving

This is the code for paper "[MSSF: A 4D Radar and Camera Fusion Framework With Multi-Stage Sampling for 3D Object Detection in Autonomous Driving](https://ieeexplore.ieee.org/document/10947638)" (T-TITS 2025). 

MSSF is a simple but effective multi-stage sampling fusion (MSSF) network based on 4D radar and camera for 3D object detection.
![image](asserts/overall.png) 

## 🔧 Install
You can prepare environment via conda:
```bash
conda env create -f environment.yml -n mssf
```
Clone our repository and install (based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)):
```bash
pip install -e . -v
```

## 📚 Dataset Preparation

### 1. Download Datasets
We use [VoD](https://github.com/tudelft-iv/view-of-delft-dataset) and [TJ4DRadSet](https://github.com/TJRadarLab/TJ4DRadSet) in our experiments. Please download the dataset and put them in `data`. We organize data as follows:
```bash
data
 |
 |-- vod
 |     |—- ImageSets
 |     |-- testing
 |     |-- training
 |             |
 |             |-- calib
 |             |-- image_2
 |             |-- label_2
 |             |-- pose
 |             |-- velodyne
 |--- tj4d
 |     |—— ImageSets
 |     |-- testing
 |     |-- training
 |              |-- calib
 |              |-- image_2
 |              |-- label_2
 |              |-- velodyne
```

### 2. Generate pickle info and reduced points (in FoV)
First, we should generate pickle info and reduced points in FoV, which is common process in `mmdetecton3d` and `OpenPCDet` codebases.
```bash
VoD dataset: python tools/create_data.py --dataset vod --root-path ./data/vod --out-dir ./data/vod
TJ4D dataset: python tools/create_data.py --dataset tj4d --root-path ./data/tj4d --out-dir ./data/tj4d
```

## 🚀 Run

### Checkpoint
Please download [checkpoints & configs](https://huggingface.co/EricLiuhhh/MSSF/tree/main), and put them in `exps`.

### Evaluation

To run evaluation:
```bash
python tools/test.py $CONFIG --checkpoint $CHECKPOINT --metric [VoDMetric,TJ4DMetric]
```

#### VoD Dataset
|Method|mAP @ EAA| mAP @ DC |
|------|------|------|
| MSSF-V | 59.96 | 81.32 |
| MSSF-PP | 63.31 | 79.78 |

#### TJ4DRadSet Dataset
|Method|mAP 3D| mAP BEV |
|------|------|------|
| MSSF-V | 37.97 | 43.11 |
| MSSF-PP | 41.75 | 48.41 |

### Training

## 🙏🏻 Acknowledgement
[MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet.git), [VoD dataset](https://github.com/tudelft-iv/view-of-delft-dataset), [TJ4DRadSet dataset](https://github.com/TJRadarLab/TJ4DRadSet)

## ⭐️ Citation
```bibtex
@ARTICLE{10947638,
  author={Liu, Hongsi and Liu, Jun and Jiang, Guangfeng and Jin, Xin},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={MSSF: A 4D Radar and Camera Fusion Framework With Multi-Stage Sampling for 3D Object Detection in Autonomous Driving}, 
  year={2025},
  volume={26},
  number={6},
  pages={8641-8656},
  doi={10.1109/TITS.2025.3554313}}
```

