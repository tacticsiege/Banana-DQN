import numpy as np
import time

from unityagents import UnityEnvironment
from agent_utils import env_initialize, env_reset, state_reward_done_unpack

from dqn_agent import DQN_Agent
from agent_utils import load_dqn

from agent_utils import load_params, load_weights

def demo_agent(env, agent, n_episodes, epsilon=0.05, seed=0, train_mode=False):
    print(f'\r\nRunning demo of \'{agent.name}\' with epsilon={epsilon}')
    scores = []
    for i in range(1, n_episodes+1):
        score = 0
        state = env_reset(env, agent.brain_name, train_mode=train_mode)
        while True:
            action = int(agent.act(state, epsilon))
            env_info = env.step(action)[agent.brain_name]
            next_state, reward, done = state_reward_done_unpack(env_info)
            
            score += reward
            state = next_state
            if done:
                break
        scores.append(score)
        print(f'Episode {i}\tScore: {score:.2f}')

    print('\r\nDemo complete! Scores:\tMin:{:.2f}\tMax:{:.2f}\tAvg:{:.3f}'.format(
        np.min(scores), np.max(scores), np.mean(scores)))
    return scores

def demo_saved_agent(env, agent_name, n_episodes=3, epsilon=0.05, seed=0,
                     train_mode=False, verbose=False):
    # initialize environment and scenario info
    brain, brain_name, state, action_size, state_size = env_initialize(env, train_mode=train_mode)
    
    # load the agent params and create the agent
    params, local_weights, target_weights = load_dqn(agent_name, verbose=verbose)    
    agent = DQN_Agent(state_size, action_size, brain_name, seed, params=params)
    print(agent.display_params())
    
    # set trained agent weights
    agent.qnetwork_local.load_state_dict(local_weights)
    agent.qnetwork_target.load_state_dict(target_weights)
    
    # run demo
    return demo_agent(env, agent,
                    n_episodes=n_episodes, epsilon=epsilon, 
                    seed=seed, train_mode=train_mode)

def demo_random_agent_discrete(env, n_episodes=3, train_mode=False, verbose=False):
    """ Runs the environment using a uniform random action selection policy. """
    # setup the environment and get initial info
    brain, brain_name, state, action_size, state_size = env_initialize(env, train_mode=train_mode, verbose=verbose)
    
    start_time = time.time()
    for n_episode in range(1, n_episodes+1):
        # reset the environment for the new episode
        state = env_reset(env, brain_name, train_mode=train_mode)
        
        # track scores and the number of steps in an episode
        score = 0
        steps = 0
        
        while True:
            # choose a random action
            action = np.random.randint(action_size)
            
            # send action to environment and get updated info
            env_info = env.step(action)[brain_name]
            next_state, reward, done = state_reward_done_unpack(env_info)
            score += reward
            steps += 1
            
            # set the state for next iteration
            state = next_state
            if done:
                break # end episode if we get the done signal
        
        print (f'Episode {n_episode} score: {score} in {steps} steps.')
        
    end_time = time.time()
    avg_episode_time = (end_time - start_time) / n_episodes
    print (f'Random agent demo complete, avg episode duration: {avg_episode_time:.3f}s.')