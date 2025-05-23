<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="data/camma_logo_tr.png" width="30%">
</a>
</div>

## **When do they StOP?: A First Step Towards Automatically Identifying Team Communication in the Operating Room**
Keqi Chen, Lilien Schewski, [Vinkle Srivastav](https://vinkle.github.io/), Joël Lavanchy, Didier Mutter, Guido Beldi, Sandra Keller*, Nicolas Padoy*, IPCAI 2025

*Co-last authors

[![arXiv](https://img.shields.io/badge/arxiv-2502.08299-red)](https://arxiv.org/abs/2502.08299)

## Introduction

**Purpose**: Surgical performance depends not only on surgeons' technical skills but also on team communication within and across the different professional groups present during the operation. Therefore, automatically identifying team communication in the OR is crucial for patient safety and advances in the development of computer-assisted surgical workflow analysis and intra-operative support systems. To take the first step, we propose a new task of detecting communication briefings involving all OR team members, i.e. the team Time-out and the StOP?-protocol, by localizing their start and end times in video recordings of surgical operations.

**Methods**: We generate an OR dataset of real surgeries, called Team-OR, with more than one hundred hours of surgical videos captured by the multi-view camera system in the OR. The dataset contains temporal annotations of 33 Time-out and 22 StOP?-protocol activities in total. We then propose a novel group activity detection approach, where we encode both scene context and action features, and use an efficient neural network model to output the results.

**Results**: The experimental results on the Team-OR dataset show that our approach outperforms existing state-of-the-art temporal action detection approaches. It also demonstrates the lack of research on group activities in the OR, proving the significance of our dataset. 

**Conclusion**: We investigate the Team Time-Out and the StOP?-protocol in the OR, by presenting the first OR dataset with temporal annotations of group activities protocols, and introducing a novel group activity detection approach that outperforms existing approaches.

### Overall framework
<p float="center"> <img src="data/teamor.gif" width="100%" /> </p>

#### In this repo we provide:
- Training and testing code for the group activity detection approach in the OR. 

## Installation
1. Please ensure that you have installed PyTorch and CUDA. **(This code requires PyTorch version >= 1.11. We use
   version=1.13.1 in our experiments)**

2. Install the required packages by running the following command:

```shell
pip install  -r requirements.txt
```

3. Install NMS

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

4. Done! We are ready to get start!

## Data preparation

We are currently unable to make Team-OR publicly available due to strict privacy and ethical concerns. If you want to train on your own data, you can conduct the following steps:

1. Feature extraction. We utilize [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2), [mmpose](https://github.com/open-mmlab/mmpose) and [mmaction2](https://github.com/open-mmlab/mmaction2) for feature extraction. 
2. Write your own dataset class under `./libs/datasets/` and import it in `./libs/datasets/__init__.py`. 
3. Write the config file under `./configs/`. 

## Training
```
python train.py ./configs/team_or.yaml --output pretrained
```

## Testing
```
python eval.py ./configs/team_or.yaml ckpt/team_or_pretrained/epoch_059.pth.tar
```

### References
The project uses [TriDet](https://github.com/dingfengshi/TriDet) and [TemporalMaxer](https://github.com/TuanTNG/TemporalMaxer). We thank the authors for releasing their codes. If you use TriDet or TemporalMaxer, consider citing it using the following BibTeX entry.
```bibtex
@inproceedings{TriDet,
    title     = {Tridet: Temporal action detection with relative boundary modeling},
    author    = {Shi, Dingfeng and Zhong, Yujie and Cao, Qiong and Ma, Lin and Li, Jia and Tao, Dacheng},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages     = {18857--18866},
    year      = {2023}
}
```
```bibtex
@article{TemporalMaxer,
    title     = {Temporalmaxer: Maximize temporal context with only max pooling for temporal action localization},
    author    = {Tang, Tuan N and Kim, Kwonyoung and Sohn, Kwanghoon},
    journal   = {arXiv preprint arXiv:2303.09055},
    year      = {2023}
}
```

The project also leverages following research works. We thank the authors for releasing their codes.
- [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2)
- [mmpose](https://github.com/open-mmlab/mmpose)
- [mmaction2](https://github.com/open-mmlab/mmaction2)

## License
This code is available for non-commercial scientific research purposes as defined in the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.
