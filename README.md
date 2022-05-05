# python-vehicle-routing
Python implementation of Dynamic Vehicle Routing (DVR) algorithms

 |  Random Assignment  | First-Come-First-Serve |
 | :-----------------: | :--------------------: |
 | ![](img/random.gif) |   ![](img/fcfs.gif)    |

|    m-SQM Policy    | UTSP (queue size = 5) |
| :----------------: | :-------------------: |
| ![](img/m_sqm.gif) |   ![](img/utsp.gif)   |

| m-DC Policy (distributed partitioning) | No-Communication (NC) Policy |
| :------------------------------------: | :--------------------------: |
|      ![](img/dc_partitioning.gif)      |       ![](img/nc.gif)        |

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

3. Some arguments to the main function can be viewed by
```bash
python src/main.py --help
```
which would return
```bash
usage: main.py [-h] time mode

positional arguments:
  time
  mode

optional arguments:
  -h, --help  show this help message and exit
```
where *time* specifies the total expected simulation time, and *mode* specifies the policy to use.

## Overview
This repo reviews the following DVR algorithms:
* Random assignment (vanilla)
* First-Come-First-Serve (FCFS)
* m-SQM (Stochastic Queue Median) policy
* UTSP (Unbiased Traveling Salesman Problem) policy
* m-DC (Divide and Conquer) Policy
* NC (No Communication) Policy

## Known Issues
1. When `time` is specified before starting the simulation, an error would occur when the simulation stops:
```bash
Traceback (most recent call last):
  File "/Users/zhihaoruan/miniforge3/envs/python_dvr/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 1198, in _on_timer
    ret = func(*args, **kwargs)
  File "/Users/zhihaoruan/miniforge3/envs/python_dvr/lib/python3.8/site-packages/matplotlib/animation.py", line 1415, in _step
    self.event_source.interval = self._interval
AttributeError: 'NoneType' object has no attribute 'interval'
```
This is due to the calling of `plt.close()` on the plt.FuncAnimation object while the animation is still running in the backend. There has been no quick fix so far, and this error can be safely ignored. 

## References
- Visualization framework from MEAM 624: Distributed Robotics, University of Pennsylvania, Spring 2022.
- F. Bullo, E. Frazzoli, M. Pavone, K. Savla and S. L. Smith, "[Dynamic Vehicle Routing for Robotic Systems](https://ieeexplore.ieee.org/abstract/document/5954127?casa_token=sAaSTkWYbO8AAAAA:eE9HJHY242a0InCpEhtyF0-iPnP2DSIq73AVHbDkbQVy-yuM4i_RGsC-RiwneH00c-z6EfxoNdU)," in Proceedings of the IEEE, vol. 99, no. 9, pp. 1482-1504, Sept. 2011, doi: 10.1109/JPROC.2011.2158181.