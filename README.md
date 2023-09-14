# Discrete
- [X] Deep Q learning (DQL)
- [X] Double Deep Q learning (DDQN)
- [X] Stochastic Actor-Critic (AC)
- [ ] Soft Actor-Critic (SAC)
- [X] Advantage Actor-Critic (A2C)

# Continuous
- [X] Stochastic Actor-Critic (AC)
- [X] Deep Deterministic Policy Gradient (DDPG) 
- [X] Advantage Actor-Critic (A2C)
- [X] Soft Actor-Critic (SAC)
- [X] Twin Delayed Deep Deterministic Policy Gradient (TD3)
- [X] Proximal Policy Optimization (PPO)

# Information

This repository encompasses implementations of RL algorithms tailored for both continuous and discrete environments, as delineated above. The codebase adheres as closely as feasible to the original papers, and supplementary enhancements drawn from other repositories are used. The appended table provides a succinct overview of key attributes characterizing the developed algorithms. Specifically, 'AV' signifies action-value, 'SV' designates state-value, 'Dt' conveys deterministic, and 'St' denotes stochastic.

|   | DDPG  | TD3  | A2C  | SAC  | PPO  |
|---|---|---|---|---|---|
| Topology  | AV  | AV  | SV  | SV+AV  | AV  |
| Action  |  Dt | Dt  | St  | St  | Dt  |
| Replay Buffer  | &check;  | &check; |  &#9746;  | &check;   | &check;  |
| Policy  | Off  | Off  | On  | Off  | Off  |
| Advantage Func.  |   &#9746; |  &#9746;  | &check;   | entropy-based  |  &check;   |
