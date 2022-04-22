# python-vehicle-routing
Python implementation of Dynamic Vehicle Routing (DVR) algorithms

|    m-SQM Policy    | First-Come-First-Serve |
| :----------------: | :--------------------: |
| ![](img/m_sqm.gif) |   ![](img/fcfs.gif)    |

| Divide-and-Conquer |  Random Assignment  |
| :----------------: | :-----------------: |
|  ![](img/dc.gif)   | ![](img/random.gif) |

## Supported Platform:
- Anaconda on Ubuntu 20.04 LTS
- Miniforge on Apple M1 Mac

## Usage
1. Install required dependency:
```bash
conda env create -f environment.yml
```

2. Execute main script:
```bash
conda activate python_dvr
python src/main.py
```

## Overview
This repo reviews the following DVR algorithms:
* Random assignment (vanilla)
* First-Come-First-Serve (FCFS)
* m-SQM
* Divide-and-Conquer (DC)

## References
- F. Bullo, E. Frazzoli, M. Pavone, K. Savla and S. L. Smith, "[Dynamic Vehicle Routing for Robotic Systems](https://ieeexplore.ieee.org/abstract/document/5954127?casa_token=sAaSTkWYbO8AAAAA:eE9HJHY242a0InCpEhtyF0-iPnP2DSIq73AVHbDkbQVy-yuM4i_RGsC-RiwneH00c-z6EfxoNdU)," in Proceedings of the IEEE, vol. 99, no. 9, pp. 1482-1504, Sept. 2011, doi: 10.1109/JPROC.2011.2158181.