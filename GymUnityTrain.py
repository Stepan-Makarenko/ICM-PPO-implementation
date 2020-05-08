import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from tensorboardX import SummaryWriter
from utils import Memory
from ICMPPO import ICMPPO
import torch.nn as nn
from torch.distributions import Categorical

from gym_unity.envs import UnityEnv

render = False
solved_reward = 1.7     # stop training if avg_reward > solved_reward
log_interval = 1000     # print avg reward in the interval
max_episodes = 350      # max training episodes
max_timesteps = 1000    # max timesteps in one episode
update_timestep = 2048  # Replay buffer size, update policy every n timesteps
log_dir= './'           # Where to store tensorboard logs

# Initialize Unity env
multi_env_name = './envs/Pyramid/Unity Environment.exe'
multi_env = UnityEnv(multi_env_name, worker_id=0,
                     use_visual=False, multiagent=True)

# Initialize log_writer, memory buffer, icmppo
writer = SummaryWriter(log_dir)
memory = Memory()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = ICMPPO(writer=writer, device=device)

timestep = 0
T = np.zeros(16)
state = multi_env.reset()
# training loop
for i_episode in range(1, max_episodes + 1):
    episode_rewards = np.zeros(16)
    episode_counter = np.zeros(16)
    for i in range(max_timesteps):
        timestep += 1
        T += 1
        # Running policy_old:
        actions = agent.policy_old.act(np.array(state), memory)
        state, rewards, dones, info = multi_env.step(list(actions))

        # Fix rewards
        dones = np.array(dones)
        rewards = np.array(rewards)
        rewards += 2 * (rewards == 0) * (T < 1000)
        episode_counter += dones
        T[dones] = 0
        # Saving reward and is_terminal:
        memory.rewards.append(rewards)
        memory.is_terminals.append(dones)

        # update if its time
        if timestep % update_timestep == 0:
            agent.update(memory, timestep)
            memory.clear_memory()

        episode_rewards += rewards

    if episode_counter.sum() == 0:
        episode_counter = np.ones(16)

    # stop training if avg_reward > solved_reward
    if episode_rewards.sum() / episode_counter.sum() > solved_reward:
        print("########## Solved! ##########")
        writer.add_scalar('Mean_extr_reward_per_1000_steps',
                          episode_rewards.sum() / episode_counter.sum(),
                          timestep
        )
        torch.save(agent.policy.state_dict(), './ppo.pt')
        torch.save(agent.icm.state_dict(), './icm.pt')
        break

    # logging
    if timestep % log_interval == 0:
        print('Episode {} \t episode reward: {} \t'.format(i_episode, episode_rewards.sum() / episode_counter.sum()))
        writer.add_scalar('Mean_extr_reward_per_1000_steps',
                          episode_rewards.sum() / episode_counter.sum(),
                          timestep
        )