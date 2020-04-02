import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import importlib
import model
importlib.reload(model)
import buffer
importlib.reload(buffer)


class SACAgent:

    def __init__(self, env, hyperparams):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        self.action_size = brain.vector_action_space_size
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        self.state_size = states.shape[1]

        # hyperparameters
        self.gamma = hyperparams["gamma"]
        self.tau = hyperparams["tau"]
        self.update_step = 0
        self.delay_step = 2

        # initialize networks
        self.q_net1 = network.QNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.q_net2 = network.QNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.target_q_net1 = network.QNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.target_q_net2 = network.QNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.policy_net = network. GaussianPolicyNetwork(self.state_size, self.action_size, hyperparams).to(self.device)

        # copy params to target param
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)


        # initialize optimizers
        q_learn_rate = hyperparams["q_learn_rate"]
        policy_learn_rate = hyperparams["policy_learn_rate"]
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_learn_rate)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_learn_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_learn_rate)

        # entropy temperature
        self.alpha = hyperparams["alpha"]
        a_learn_rate = hyperparams["a_learn_rate"]
        self.target_entropy = -brain.vector_action_space_size
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_learn_rate)

        self.replay_buffer = buffer.SimpleBuffer(self.device, 0, hyperparams)


    def learn_episode(self, max_steps):
        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        episode_reward = 0

        for step in range(max_steps):
            action = self.get_action(state)
            env_info = self.env.step(action)[brain_name]
            reward = env_info.rewards[0]
            next_state = env_info.vector_observations[0]
            done = env_info.local_done[0]
            self.replay_buffer.add(buffer.Experience(state, action, reward, next_state, done))
            episode_reward += reward

            if self.replay_buffer.ready_to_sample():
                self.update()

            if done or step == max_steps-1:
                break

            state = next_state

        return episode_reward

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return action

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        dones = dones.view(dones.size(0), -1)

        next_actions, next_log_pi = self.policy_net.sample(next_states)
        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update q networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # delayed update for policy network and target q networks
        new_actions, log_pi = self.policy_net.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions)
            )
            policy_loss = (self.alpha * log_pi - min_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # target networks
            for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1
