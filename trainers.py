# req agent_utils, dqn_agent, dqn_model
import numpy as np
import time
import os
from collections import namedtuple, deque

import torch

from agent_utils import env_initialize, env_reset, state_reward_done_unpack
from agent_utils import save_dqn



# training methods
def train_dqn(env, agent,
                n_episodes=1000, 
                goal_score=15, keep_training=False,
                eps_start=1.0, eps_end=0.001, eps_decay=0.995,
                score_window_size=100, print_every=100
                ):
    is_solved = False
    scores = []
    scores_window = deque(maxlen=score_window_size) # last 100 episode scores
    duration_window = deque(maxlen=5) # last 5 episode durations
    
    # save agent params and prepare storage for results
    save_dqn(agent.name, params=agent.params, verbose=True)
    
    # initialize epsilon
    epsilon_svc = EpsilonService(
        method='decay', start_value=eps_start, end_value=eps_end, 
        decay_rate=eps_decay, n_episodes=n_episodes)
    epsilon = epsilon_svc.get_value()
    
    print(f'\r\nTraining started for \'{agent.name}\'...')
    training_start_time = time.time()
    for i_episode in range(1, n_episodes+1):
        # reset for new episode        
        score = 0
        state = env_reset(env, agent.brain_name, train_mode=True)
        
        # run episode
        episode_start_time = time.time()
        while True:
            # choose and take action
            action = int(agent.act(state, epsilon))
            env_info = env.step(action)[agent.brain_name]
            next_state, reward, done = state_reward_done_unpack(env_info)
            
            # update agent with new state and reward
            agent.step(state, action, reward, next_state, done)
            score += reward
            
            state = next_state # update state for next timestep
            if done:
                break
        
        # decay
        epsilon = epsilon_svc.update(i_episode)
        
        # track progress
        duration = time.time() - episode_start_time
        scores.append(score)
        scores_window.append(score)
        duration_window.append(duration)
        
        # display progress and save checkpoints
        avg_score = np.mean(scores_window)
        avg_duration = np.mean(duration_window)
        print('\rEpisode {}\tAvg. Score: {:.2f}\tAvg. Duration: {:.3f}s'.format(
            i_episode, avg_score, avg_duration), end="")
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAvg. Score: {:.2f}\tAvg. Duration: {:.4f}s\tEpsilon: {:.6f}'.format(
                i_episode, avg_score, avg_duration, epsilon))
            # todo: save checkpoint weights
            save_dqn(agent.name, 
                agent.qnetwork_local.state_dict(),
                agent.qnetwork_target.state_dict(),
                scores=scores,
                i_checkpoint=i_episode)
            
        if avg_score > goal_score and i_episode > score_window_size and not is_solved:
            is_solved = True
            print('\rEpisode {}\tAvg. Score: {:.2f}\tAvg. Duration: {:.4f}s\tEpsilon: {:.6f}'.format(
                i_episode, avg_score, avg_duration, epsilon))
            print('\r\nEnvironment solved in {} episodes!'.format(i_episode))
            print('\rAverage Score for last {} episodes: {:.2f}\tGoal: {}'.format(
                score_window_size, avg_score, goal_score))
            print('\rTotal Duration: {:.2f}m\n'.format((time.time() - training_start_time)/ 60.0))
            # todo: save solved model weights and print
            save_dqn(agent.name, 
                agent.qnetwork_local.state_dict(),
                agent.qnetwork_target.state_dict(),
                scores=scores, verbose=True)
            
            if keep_training:
                print('\r\nContinuing training...')
            else:
                return scores
    
    # finished all episodes
    print('\r\nCompleted training on {} episodes.'.format(n_episodes))
    print('\rAverage Score for last {} episodes: {:.2f}\tGoal: {}'.format(
        score_window_size, np.mean(scores_window), goal_score))
    print('\rTotal Duration: {:.2f}m\n'.format((time.time() - training_start_time)/ 60.0))
    
    # save the agent
    save_dqn(agent.name, 
        agent.qnetwork_local.state_dict(),
        agent.qnetwork_target.state_dict(),
        scores=scores)

    return scores

# EpsilonService
class EpsilonService:
    """ Maintains epsilon value for epsilon-greedy policy in training. """
    def __init__(self, method='decay', start_value=1.0, end_value=0.001, decay_rate=0.995, n_episodes=1000):
        
        # control values for epsilon
        self.start_value = start_value
        self.end_value = end_value
        self.current_value = start_value
        
        # method specific params
        self.method = method
        self.decay_rate = decay_rate # decay rate for decay method
        self.n_episodes = n_episodes # max episodes for linear method
        
    def reset(self, method=None, start_value=None, end_value=None, decay_rate=None, n_episodes=None):
        # apply any provided changes
        self.update_settings_(method, start_value, end_value, decay_rate, n_episodes)
            
        # reset and return the epsilon value
        self.current_value = self.start_value
        return self.current_value
        
    def update_settings_(self, method=None, start_value=None, end_value=None, decay_rate=None, n_episodes=None):
        # update any settings if provided
        if method is not None:
            self.method = method
        if start_value is not None:
            self.start_value = start_value
        if end_value is not None:
            self.end_value = end_value
        if decay_rate is not None:
            self.decay_rate = decay_rate
        if n_episodes is not None:
            self.n_episodes = n_episodes
        
    def update(self, i=None):
        if self.method == 'decay':
            self.current_value = max(self.end_value, self.current_value * self.decay_rate)
        elif self.method == 'linear' and i is not None:
            self.current_value = np.interp([i], [1, self.n_episodes], 
                                           [self.start_value, self.end_value], 
                                           right=self.end_value)[0]
        else:
            print (f'Unknown epsilon update method: {self.method}, i={i}')
        
        return self.current_value
    
    def get_value(self):
        return self.current_value
