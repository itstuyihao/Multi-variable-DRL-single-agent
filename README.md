# Multi-Variable Single-Agent DRL Model for CW Optimization

## Overview
This repository implements the **multi-variable DRL model** proposed in the paper “Enhanced-SETL: A multi-variable deep reinforcement learning approach for contention window (CW) optimization in dense Wi-Fi networks.”(https://doi.org/10.1016/j.comnet.2024.110690) The script in `main.py` builds a Double DQN (PARL/PaddlePaddle) agent that jointly tunes the CW minimum (`CWmin`) and CW threshold (`CWThreshold`) value to stabilize throughput and fairness for IEEE 802.11 DCF. The environment is simplified (no channel fading or PHY timing beyond slot-level events) and is intended for algorithm verification, not standard-compliant performance claims.

## Requirements
- Python ≥ 3.8
- Python packages: `paddlepaddle` (fluid API), `parl`, `numpy`
- Optional: `matplotlib` (only if you add your own plotting)

## Repository Structure
```
multi-variable-drl/
├── README.md
├── main.py                     # DDQN agent adjusting CWmin & CW threshold via slot-level simulation
└── data/50000_simtime/10_nodes/# create before running; receives logs
    ├── output.txt              # reward trace (time, normalized throughput)
    ├── loss.txt                # training loss over time
    ├── thr.txt                 # instantaneous throughput
    └── latency.txt             # average packet latency per log interval
```

## Functions
- `class Model(act_dim)`: two-layer fully connected Q-network (128/128 ReLU) outputting action values.
- `class DDQN(model, act_dim, gamma, lr)`: Double DQN with target network; `learn()` uses Adam, `sync_target()` copies weights.
- `class Agent(algorithm, obs_dim, act_dim, e_greed, e_greed_decrement)`: epsilon-greedy action selection (`sample`), greedy inference (`predict`), and periodic target update every 200 steps.
- `class ReplayMemory(max_size)`: deque-based buffer with uniform `sample(batch_size)`.
- `def new_resolve(new_cwmin, new_cwthreshold)`: executes one contention episode using current CW settings, updates backoff counters, success/collision stats, and time progression.
- `def main()`: sets simulation constants (10 stations, 50,000 s), warms up memory, trains online with 9 discrete actions (ΔCWmin × ΔCWthreshold), logs reward/loss/throughput/latency, then runs a short greedy evaluation.
- `def printStats()` / `printLatency()`: report aggregate throughput, collision rate, Jain’s fairness index, and mean latency.

## Quick start
From the repo root:
```bash
cd multi-variable-drl
python3 -m pip install paddlepaddle parl numpy
mkdir -p data/50000_simtime/10_nodes
python3 main.py
```
Adjust simulation parameters (e.g., `_n`, `_simTime`, `_cwmin`, `_cwmax`, `_pktSize`) at the top of `main.py` as needed for your experiments.

## Cite
If you use this code, please cite:
```
@article{TU2024110690,
author = {Tu, Yi-Hao and Lin, En-Cheng and Ke, Chih-Heng and Ma, Yi-Wei},
journal = {Computer Networks},
title = {Enhanced-SETL: A multi-variable deep reinforcement learning approach for contention window optimization in dense Wi-Fi networks},
year = {2024},
volume = {253},
pages = {110690},
keywords = {Enhanced-SETL, Contention window, Fairness index, Wireless networks},
doi = {https://doi.org/10.1016/j.comnet.2024.110690}
}
```

## Contact
Open an issue or reach out to the author (itstuyihao@gmail.com).
