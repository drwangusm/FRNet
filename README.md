# FRNet

Code for paper: [Enhancing Classroom Behavior Recognition with Lightweight Multi-Scale Feature Fusion](https://www.techscience.com/cmc/online/detail/23901)

# Requirements
  ![Python >=3.8.16](https://img.shields.io/badge/Python->=3.8.16-yellow.svg)    ![torch >=2.2.2](https://img.shields.io/badge/Pytorch->=2.2.2-blue.svg)

```
 conda create -n yolov11 python=3.8.16

 conda activate yolov11

 pip install torch==2.2.2+cu121
```

# Dataset
The datasets we used are as follows:
- [STBD-08](https://ieeexplore.ieee.org/abstract/document/10185142)
- [SCB-dataset3](https://link.springer.com/chapter/10.1007/978-3-031-46311-2_4)

The source dataset was downloaded from the dataset github page https://github.com/Whiffe/SCB-dataset (Accessed on 1th Nov 2024).

Due to database requirements in this paper, we provide forever restricted links access to the reprocessed dataset :
- [STBD-08](https://pan.baidu.com/s/1p9yygeBTTutSykQpaoVZ2Q?pwd=254u)
- [SCB-dataset3](https://pan.baidu.com/s/18VymPILVSuXFrtv3s6I8-Q?pwd=fdcb)

Thanks to the authors for providing the open access STBD-08 and SCB-dataset3 dataset.

# Running
## Modify configs
You can modify the configuration of the parameters in the XXX.yaml for different dataset.

## Training
You can run the bash script as below :
```bash
python train.py

```
Models and results will be saved at folder: 'runs/dataset_name/'. 

## Results
The results of the trained model can be downloaded directly from this URL:[FRNet/runs](https://pan.baidu.com/s/1e3CSHxpqgf3Dq7vjaquwGw?pwd=jet2)

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@Article{cmc.2025.066343,
AUTHOR = {Chuanchuan Wang, Ahmad Sufril Azlan Mohamed, Xiao Yang , Hao Zhang , Xiang Li, Mohd Halim Bin Mohd Noor},
TITLE = {Enhancing Classroom Behavior Recognition with Lightweight Multi-Scale Feature Fusion},
JOURNAL = {Computers, Materials \& Continua},
VOLUME = {85},
YEAR = {2025},
NUMBER = {1},
PAGES = {855--874},
URL = {http://www.techscience.com/cmc/v85n1/63535},
ISSN = {1546-2226},
DOI = {10.32604/cmc.2025.066343}
}
```

