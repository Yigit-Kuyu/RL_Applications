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


&emsp;




| Algorithm | Component           | Equation |
|-----------|---------------------|----------|
| SAC       | Objective Function  | $$J(\pi) = E_{(s_t, a_t) \sim \rho_\pi}[r(s_t, a_t) + \alpha H(\pi(\cdot\|s_t))]$$ |
|           | Critic Update       |  $$Q_{\text{target}} = r + \gamma (1-d) [ \min_{i=1,2} Q_{\text{target},i}(s', \tilde{a}') - \alpha \log \pi(\tilde{a}'\|s')]$$ |
|           | Actor Update        | $$\nabla_\theta J(\pi_\theta) = E_{s, a \sim \mathcal{D}} [ \nabla_\theta \log \pi_\theta(a\|s) (\min_{i=1,2} Q_{\theta_i}(s,a) - \alpha \log \pi_\theta(a\|s)) ]$$ |
|           | Temperature Update  | Adjusts $\alpha$ based on the predefined objective |
| A2C       | Objective Function  | $$J(\theta) = E_{\pi_\theta}[\log \pi_\theta(a_t\|s_t) A_t]$$ |
|           | Policy (Actor) Update | $$\theta \leftarrow \theta + \eta \nabla_\theta \log \pi_\theta(a_t\|s_t) A(s_t, a_t)$$ |
|           | Value Function (Critic) Update | $$V(s) \leftarrow V(s) + \beta (R_t - V(s))$$ |
| PPO       | Objective Function  | $L^{CLIP}(\theta) = E_t[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]$ |
|           | Policy Update       | $\theta \leftarrow \theta + \eta \nabla_\theta \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)$ |
|           | Value Function Update | The value function is updated by minimizing the mean squared error between the predicted value and the computed returns. |
| TD3       | Objective Function  | Uses the Q-function's objective, minimized over two Q-functions to mitigate overestimation. |
|           | Critic Update       | For each of the two Q-functions: $$Q_{\text{target}} = r + \gamma \min_{i=1,2} Q_{\text{target},i}(s', \mu_{\theta'}(s' + \epsilon))$$, where $\epsilon$ is clipped noise. |
|           | Actor Update        | $$\nabla_{\theta_{\mu}} J \approx E(s \sim D) [ \nabla_a Q_{\theta_1}(s, a)  \nabla_{\theta_{\mu}}\mu_{\theta}(s) ]$$ |
| DDPG      | Objective Function  | $$J(\pi) = E_{s \sim \rho^\beta}[Q(s, \pi(s))]$$ |
|           | Critic Update       |$$L = \frac{1}{N} \sum_{i} [y_i - Q(s_i, a_i)]^2$$, where $$y_i = r_i + \gamma Q'(s_{i+1}, \pi'(s_{i+1};\theta^{\pi'}))$$ |
|           | Actor Update        | $$\nabla_{\theta^\pi} J \approx (1/N) \sum_{i} ( \nabla_a Q(s, a;\theta^Q) \nabla_{\theta^\pi} \pi(s;\theta^\pi) )$$ |

