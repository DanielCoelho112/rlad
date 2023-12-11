# RLAD
This is implementation of RLAD, which is described in:

> **RLAD: Reinforcement Learning from Pixels for Autonomous Driving in Urban Environments**
>
> [Daniel Coelho](https://anthonyhu.github.io/), 
[Miguel Oliveira](https://github.com/gianlucacorrado),
[Vítor Santos](https://github.com/nicolasgriffiths).
>
> [IEEE Transactions on Automation Science and Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8856)<br/>

If you find our work useful, please consider citing: WIP
<!-- ```bibtex
@inproceedings{mile2022,
  title     = {Model-Based Imitation Learning for Urban Driving},
  author    = {Anthony Hu and Gianluca Corrado and Nicolas Griffiths and Zak Murez and Corina Gurau
   and Hudson Yeo and Alex Kendall and Roberto Cipolla and Jamie Shotton},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  year = {2022}
} -->

## Setup
- Clone the repository with `git clone git@github.com:DanielCoelho112/rlad.git`
- Download [CARLA 0.9.10.1](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1).
- Run the docker container with `docker run -it --gpus all --network=host -v results_path:/root/results/rlteacher -v rlad_path:/root/rlad danielc11/rlad:0.0 bash`
where `results_path` is the path where the results will be written, and `rlad_path` is the path of the rlad repository.


## Training
- Start the CARLA server
- Run: python3 rlad/run/python3 main.py -en rlad_original


## Credits
Thanks to the authors of [End-to-End Urban Driving by Imitating a Reinforcement Learning Coach](https://github.com/zhejz/carla-roach)
for providing a framework to train RL agent in CARLA.