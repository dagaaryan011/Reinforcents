# Reinforcement Learning:
Reinforcement learning is a process where an agent learns by interacting with an environment. It takes actions and gets rewards or penalties based on those actions. Over time, the agent improves its decisions to maximize the total rewards it receives.
![rl](img/rl.jpg)

# Reinforcement learning algorithms used:

## 1. SAC (Soft - Actor Critic)
SAC uses: <br>
**1. Actor Network**
Chooses actions by sampling from a learned probability distribution. <br>
**2. Critic Networks (2x)**
Evaluate how good a specific action is in a given state by estimating its expected future reward (Q-value); two are used for stability. <br>
**3. Value Network**
Estimates how good a state is overall, assuming the agent follows its current policy, regardless of specific actions. <br>
**4. Target Value Network**
A slowly-updated copy of the value network that provides stable targets during training to prevent instability. <br><br>

SAC maximizes the expected reward plus an entropy term to encourage exploration. <br>
### Entropy term:
Let x be a random variable with probability mass or density function P. The entropy H of x is computed from its distribution P according to: <br>
![entropy](img/SAC_entropy.png) <br><br>
The RL problem changes to: <br>
![RL problem](img/SAC_RLprblm.png) <br><br>
### Actor update :<br>
![actor loss](img/SAC_actor_loss.png) <br><br>

### Critic update: <br>
![critic loss](img/SAC_critic_loss.png) <br><br>

### Value update:<br>
![value loss](img/SAC_value_loss.png) <br>
The minimum of the 2 critics values is taken <br><br>

### Target value soft update: <br>
![target loss](img/SAC_target_loss.png)<br><br>



## 1. DDPG (Deep Deterministic Policy Gradient)
DDPG uses: <br>
**1. Actor Network**
A deterministic policy network that outputs the exact action to take in a given state. It aims to maximize the expected Q-value as judged by the critic. <br>
**2. Critic Network**
Estimates the Q-value: how good a given action is in a specific state under the current policy. <br>
**3. Target Actor Network**
A delayed, slowly-updated copy of the actor network used to generate target actions during critic updates. It improves stability by avoiding rapid policy changes. <br>
**4. Target Critic Network**
A slowly-updated copy of the critic network used to compute stable Q-targets when training the main critic. <br>

### Target for critic:
![critic target](img/DDPG_critic_target.png)

### Critic update: <br>
![critic loss](img/DDPG_critic_loss.png)

### Actor update: <br>
![actor loss](img/DDPG_actor_loss.png)

### Soft actor and critic update: <br>
![soft update](img/DDPG_soft_update.png)







