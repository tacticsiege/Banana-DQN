import numpy as np
import random
import time

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn_model import DQNetwork
from replay_buffer import UniformReplayBuffer

from util import env_initialize, env_reset, state_reward_done_unpack
from util import TrainingMonitor, EpsilonService

# default parameter values
LAYER_SIZES = [64, 64]  # size of hidden layers for DQNetwork
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size for experience sampling
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters, [0=Target, 1=Local]
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the local network
COPY_EVERY = 4          # how often to copy weights to target network

###############################################################################
###############################################################################
class DQN_Params:
    """ Parameter set used for DQN agents. """
    def __init__(self, name=None,
                layers=LAYER_SIZES, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                update_every=UPDATE_EVERY, copy_every=COPY_EVERY,
                learning_rate=LR, gamma=GAMMA, tau=TAU):
        # DQNetwork params
        self.name = name
        self.Layers = layers
        
        # replay buffer params
        self.BufferSize = buffer_size
        self.BatchSize = batch_size
        
        # update and copy rates
        self.UpdateEvery = update_every
        self.CopyEvery = copy_every

        self.LearningRate = learning_rate   # learning rate (alpha)
        self.Gamma = gamma                  # discount factor (gamma)
        self.Tau = tau                      # copy interpolation (tau)
    
    def display_params(self, condensed=True):
        p = 'h{}, exp[{}, {}], u,c[{}, {}], g,t,lr[{}, {}, {}]'.format(
            self.Layers, self.BufferSize, self.BatchSize, self.UpdateEvery, 
            self.CopyEvery, self.Gamma, self.Tau, self.LearningRate)
        if condensed:
            return p
        # todo: nicer format and use appropriate toString/repr
        return p

class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, 
                    params=DQN_Params(), verbose=False, device=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # implementation details and identity
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.description = 'DQN({})'.format(params.display_params())
        self.name = params.name if params.name is not None else self.description

        # set environment information
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DQNetwork(state_size, action_size, seed, layers=params.Layers).to(self.device)
        self.qnetwork_target = DQNetwork(state_size, action_size, seed, layers=params.Layers).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=params.LearningRate)

        # Replay memory
        self.memory = UniformReplayBuffer(action_size, params.BufferSize, params.BatchSize, seed)
        
        # Initialize time steps, t_step for updates, c_step for copying weights
        self.t_step = 0
        self.c_step = 0

        # store params for later
        self.params = params
    
    def train(self, env, n_episodes=1000, 
              eps_start=1.0, eps_end=0.001, eps_decay=0.995,
              print_every=100):
        """ Train the agent on the given established environment. """
        # monitor training progress
        monitor = TrainingMonitor(print_every=print_every)
        
        # initialize epsilon
        epsilon_svc = EpsilonService(
            method='decay', start_value=eps_start, end_value=eps_end, 
            decay_rate=eps_decay, n_episodes=n_episodes)
        eps = epsilon_svc.get_value()                   
        
        # initialize environment and obtain info
        brain, brain_name, state, action_size, state_size = env_initialize(env)
        
        # start training loop
        monitor.start()
        for i_episode in range(1, n_episodes+1):
            monitor.start_episode()
            # run episode
            score, steps, duration = self.episode(env, brain_name, eps)
            
            # track and display scores
            monitor.end_episode(i_episode, score, steps, eps)
            # update epsilon
            eps = epsilon_svc.update(i=i_episode)
        
        # finished all episodes, display results
        scores, final_avg, total_duration = monitor.end(n_episodes=i_episode)
        
        # return the unpacked scores and full episode data
        return scores, monitor.episodes

    def episode(self, env, brain_name, epsilon):
        """Run an episode. """
        # reset the environment
        score = 0
        steps = 0
        state = env_reset(env, brain_name, train_mode=True)

        start_time = time.time()
        # run episode until done
        while True:
            # choose and take action
            action = int(self.act(state, epsilon))
            env_info = env.step(action)[brain_name]
            next_state, reward, done = state_reward_done_unpack(env_info)

            # update with new state from the environment
            self.step(state, action, reward, next_state, done)
            score += reward
            steps += 1
            
            state = next_state # update state for next timestep
            if done:
                break
        
        end_time = time.time()
        duration = end_time - start_time
        return score, steps, duration
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # update timesteps
        self.t_step = (self.t_step + 1) % self.params.UpdateEvery
        self.c_step = (self.c_step + 1) % self.params.CopyEvery
        
        # only update every params.UpdateEvery timesteps
        if self.t_step == 0:
            # get random subset and learn, 
            # sample < batch_size until there are enough for a full batch
            k = min(len(self.memory), self.params.BatchSize)            
            experiences = self.memory.sample(k=k)
            self.learn(experiences, self.params.Gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # get action values for state
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model for evaluation
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Minimize the loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update the weights in the target network every c_steps
        if self.c_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.params.Tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter (0 = all target, 1 = all local)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)