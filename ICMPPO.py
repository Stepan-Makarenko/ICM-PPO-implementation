import numpy as np
import torch
import torch.nn as nn
from ICM import ICM
from ActorCritic import ActorCritic
from utils import Swish, linear_decay_beta, linear_decay_lr, linear_decay_eps


class ICMPPO:
    def __init__(self, writer, state_dim=172, action_dim=5, n_latent_var=512, lr=3e-4, betas=(0.9, 0.999),
                 gamma=0.99, ppo_epochs=3, icm_epochs=1, eps_clip=0.2, ppo_batch_size=128,
                 icm_batch_size=16, intr_reward_strength=0.02, lamb=0.95, device='cpu'):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lambd = lamb
        self.eps_clip = eps_clip
        self.ppo_epochs = ppo_epochs
        self.icm_epochs = icm_epochs
        self.ppo_batch_size = ppo_batch_size
        self.icm_batch_size = icm_batch_size
        self.intr_reward_strength = intr_reward_strength
        self.device = device
        self.writer = writer
        self.timestep = 0
        self.icm = ICM(activation=Swish()).to(self.device)

        self.policy = ActorCritic(state_dim=state_dim,
                                  action_dim=action_dim,
                                  n_latent_var=n_latent_var,
                                  activation=Swish(),
                                  device=self.device,
                                  ).to(self.device)
        self.policy_old = ActorCritic(state_dim,
                                      action_dim,
                                      n_latent_var,
                                      activation=Swish(),
                                      device=self.device
                                      ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.optimizer_icm = torch.optim.Adam(self.icm.parameters(), lr=lr, betas=betas)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss(reduction='none')

    def update(self, memory, timestep):
        # Convert lists from memory to tensors
        self.timestep = timestep
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_states = torch.transpose(old_states, 0, 1)
        old_actions = torch.stack(memory.actions).T.to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).T.to(self.device).detach()

        # Finding s, n_s, a, done, reward:
        curr_states = old_states[:, :-1, :]
        next_states = old_states[:, 1:, :]
        actions = old_actions[:, :-1].long()
        rewards = torch.tensor(memory.rewards[:-1]).T.to(self.device).detach()
        mask = (~torch.tensor(memory.is_terminals).T.to(self.device).detach()[:, :-1]).type(torch.long)
        with torch.no_grad():
            intr_reward, _, _ = self.icm(actions, curr_states, next_states, mask)
        intr_rewards = torch.clamp(self.intr_reward_strength * intr_reward, 0, 1)

        self.writer.add_scalar('Mean_intr_reward_per_1000_steps',
                               intr_rewards.mean() * 1000,
                               self.timestep
                               )

        # Finding comulitive advantage
        with torch.no_grad():
            state_values = torch.squeeze(self.policy.value_layer(curr_states))
            next_state_values = torch.squeeze(self.policy.value_layer(next_states))
            td_target = (rewards + intr_rewards) / 2 + self.gamma * next_state_values * mask
            delta = td_target - state_values

            self.writer.add_scalar('maxValue',
                                   state_values.max(),
                                   timestep
                                   )
            self.writer.add_scalar('meanValue',
                                   state_values.mean(),
                                   self.timestep
                                   )

            advantage = torch.zeros(1, 16).to(self.device)
            advantage_lst = []
            for i in range(delta.size(1) - 1, -1, -1):
                delta_t, mask_t = delta[:, i], mask[:, i]
                advantage = delta_t + (self.gamma * self.lambd * advantage) * mask_t
                advantage_lst.insert(0, advantage)

            advantage_lst = torch.cat(advantage_lst, dim=0).T
            # Get local advantage to train value function
            local_advantages = state_values + advantage_lst
            # Normalizing the advantage
            advantages = (advantage_lst - advantage_lst.mean()) / (advantage_lst.std() + 1e-10)

        # Optimize policy for ppo epochs:
        epoch_surr_loss = 0
        for _ in range(self.ppo_epochs):
            indexes = np.random.permutation(actions.size(1))
            # Train PPO and icm
            for i in range(0, len(indexes), self.ppo_batch_size):
                batch_ind = indexes[i:i + self.ppo_batch_size]
                batch_curr_states = curr_states[:, batch_ind, :]
                batch_actions = actions[:, batch_ind]
                batch_mask = mask[:, batch_ind]
                batch_advantages = advantages[:, batch_ind]
                batch_local_advantages = local_advantages[:, batch_ind]
                batch_old_logprobs = old_logprobs[:, batch_ind]

                # Finding actions logprobs and states values
                batch_logprobs, batch_state_values, batch_dist_entropy = self.policy.evaluate(batch_curr_states,
                                                                                              batch_actions)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(batch_logprobs - batch_old_logprobs.detach())

                # Apply leaner decay and multiply 16 times cause agents_batch is 16 long
                decay_epsilon = linear_decay_eps(self.timestep * 16)
                decay_beta = linear_decay_beta(self.timestep * 16)

                # Finding Surrogate Loss:
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - decay_epsilon, 1 + decay_epsilon) * batch_advantages
                loss = -torch.min(surr1, surr2) * batch_mask + \
                       0.5 * nn.MSELoss(reduction='none')(batch_state_values,
                                                           batch_local_advantages.detach()) * batch_mask - \
                       decay_beta * batch_dist_entropy * batch_mask
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                linear_decay_lr(self.optimizer, self.timestep * 16)

                epoch_surr_loss += loss.item()

        self._icm_update(self.icm_epochs, self.icm_batch_size, curr_states, next_states, actions, mask)
        self.writer.add_scalar('Lr',
                               self.optimizer.param_groups[0]['lr'],
                               self.timestep
        )
        self.writer.add_scalar('Surrogate_loss',
                               epoch_surr_loss / (self.ppo_epochs * (len(indexes) // self.ppo_batch_size + 1)),
                               self.timestep
        )

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _icm_update(self, epochs, batch_size, curr_states, next_states, actions, mask):
        epoch_forw_loss = 0
        epoch_inv_loss = 0
        for _ in range(epochs):
            indexes = np.random.permutation(actions.size(1))
            for i in range(0, len(indexes), batch_size):
                batch_ind = indexes[i:i + batch_size]
                batch_curr_states = curr_states[:, batch_ind, :]
                batch_next_states = next_states[:, batch_ind, :]
                batch_actions = actions[:, batch_ind]
                batch_mask = mask[:, batch_ind]

                _, inv_loss, forw_loss = self.icm(batch_actions,
                                                  batch_curr_states,
                                                  batch_next_states,
                                                  batch_mask)
                epoch_forw_loss += forw_loss.item()
                epoch_inv_loss += inv_loss.item()
                unclip_intr_loss = 10 * (0.2 * forw_loss + 0.8 * inv_loss)

                # take gradient step
                self.optimizer_icm.zero_grad()
                unclip_intr_loss.backward()
                self.optimizer_icm.step()
                linear_decay_lr(self.optimizer_icm, self.timestep * 16)
        self.writer.add_scalar('Forward_loss',
                               epoch_forw_loss / (epochs * (len(indexes) // batch_size + 1)),
                               self.timestep
        )
        self. writer.add_scalar('Inv_loss',
                                epoch_inv_loss / (epochs * (len(indexes) // batch_size + 1)),
                                self.timestep
        )
