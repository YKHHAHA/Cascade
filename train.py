import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # 共享特征层
        self.fc_shared = nn.Linear(state_dim, 64)

        # Actor网络(策略)
        self.actor = nn.Linear(64, action_dim)

        # Critic网络(价值)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc_shared(x))

        # 计算动作概率
        action_probs = F.softmax(self.actor(x), dim=-1)

        # 计算状态价值
        state_value = self.critic(x)

        return action_probs, state_value


# 简单的PPO实现
class PPO:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, clip_ratio=0.2):
        self.gamma = gamma  # 折扣因子
        self.clip_ratio = clip_ratio  # PPO裁剪参数

        # 创建网络和优化器
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 存储轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)

        # 获取动作概率和状态价值
        action_probs, _ = self.policy(state)

        # 创建分类分布并采样
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def update(self):
        # 将存储的数据转换为张量
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        old_log_probs = torch.stack(self.log_probs)

        # 计算折扣回报
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.FloatTensor(returns)

        # 标准化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 执行多次优化
        for _ in range(5):  # 5个优化周期
            # 获取新的动作概率和状态价值
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            # 计算优势函数
            advantages = returns - state_values.detach().squeeze()

            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # 计算替代目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages

            # 计算actor和critic损失
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns)

            # 总损失
            loss = actor_loss + 0.5 * critic_loss

            # 梯度下降
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 清空轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []


# 训练函数
def train(env_name="CartPole-v1", episodes=2000, max_steps=500):
    # 创建环境
    env = gym.make(env_name)

    # 获取状态和动作空间
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建PPO代理
    agent = PPO(state_dim, action_dim)

    # 追踪奖励
    all_rewards = []

    # 开始训练
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作
            action, log_prob = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.states.append(state)
            agent.actions.append(action)
            agent.rewards.append(reward)
            agent.dones.append(done)
            agent.log_probs.append(log_prob)

            state = next_state
            episode_reward += reward

            if done:
                break

        # 每次回合结束后更新策略
        agent.update()

        all_rewards.append(episode_reward)

        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episode {episode + 1}/{episodes}, 平均奖励: {avg_reward:.2f}")

    print("训练完成!")
    env.close()
    return all_rewards


# 运行训练
if __name__ == "__main__":
    rewards = train()
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('PPO Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
