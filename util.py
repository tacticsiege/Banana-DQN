import numpy as np
import time
from collections import namedtuple, deque

###############################################################################
# Unity environment helpers
###############################################################################
def env_initialize(env, train_mode=True, brain_idx=0, idx=0, verbose=False):
    """ Setup environment and return info  """
    # get the default brain
    brain_name = env.brain_names[brain_idx]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=train_mode)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size

    # examine the state space 
    state = env_info.vector_observations[idx]
    state_size = len(state)    
    
    if verbose:
        # number of agents in the environment
        print(f'Number of agents: {len(env_info.agents)}')
        print(f'Number of actions: {action_size}')
        print(f'States have length: {state_size}')
        print(f'States look like: {state}')
        
    return (brain, brain_name, state, action_size, state_size)

def env_reset(env, brain_name, train_mode=True, idx=0):
    """ Reset environment and get initial state """
    
    # reset the environment
    env_info = env.reset(train_mode=train_mode)[brain_name]
    state = env_info.vector_observations[idx]
    
    return state

def state_reward_done_unpack(env_info, idx=0):
    """ Unpacks the state, rewards and done signal from the environment info """
    s1 = env_info.vector_observations[idx] # get the next state
    r1 = env_info.rewards[idx]             # get the reward
    done = env_info.local_done[idx]        # see if episode has finished
    return (s1, r1, done)

def demo_random_agent(env, n_episodes=3, train_mode=True, verbose=False):
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
    print (f'Random agent demo complete, avg episode duration: {avg_episode_time}s.')

class TrainingMonitor:
    """ Tracks various training stats and timings. """
    def __init__(self, print_every=100):
        self.print_every = print_every
        self.start_time = None
        
        self._episode_start_time = None
        
        # define named tuple and create and empty set of them
        self.episode_stats = namedtuple("EpisodeStats", field_names=["i", "score", "steps", "duration", "epsilon"])
        self.episodes = []
        
        self.scores = []
        self.scores_window = deque(maxlen=100) # last 100 scores
        self.duration_window = deque(maxlen=20) # last 20 episode durations
    
    def start(self):
        self.start_time = time.time()
        
    def end(self, n_episodes):
        end_time = time.time()
        duration = end_time - self.start_time
        self.start_time = None
        
        avg_score = np.mean(self.scores_window)
        
        # display results
        self._display_complete(n_episodes, duration, avg_score)        
        
        return self.scores, avg_score, duration
        
    def start_episode(self):
        self._episode_start_time = time.time()
        
    def end_episode(self,  i, score, steps, epsilon=None):
        # track episode time
        episode_end_time = time.time()
        duration = episode_end_time - self._episode_start_time
        
        # add and display episode stats
        self._add_episode(i, score, steps, duration, epsilon)
        self._display_episode(i, np.mean(self.scores_window), np.mean(self.duration_window))
        
    def save(self, filename):
        pass
    
    def load(self, filename):
        pass
    
    def plot_scores(self, scores, fig=None, subplot=None, xlabel='Episode #', ylabel='Score'):
        # show the plot unless a figure has been provided        
        show_plot = False
        if fig is None:
            fig = plt.figure()
            subplot = 111
            show_plot = True
        
        if subplot is None:
            subplot = 111
        
        # create and plot data
        ax = fig.add_subplot(subplot)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        
        if show_plot:
            plt.show()
            
        return ax
    
    def moving_avg(self, scores, window_width=100):
        cumsum_vec = np.cumsum(np.insert(scores, 0, 0))
        avg_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        return avg_vec
        
    def _add_episode(self, i, score, steps, duration, epsilon):
        stats = self.episode_stats(i, score, steps, duration, epsilon)
        self.episodes.append(stats)
        self.scores.append(score)
        self.scores_window.append(score)
        self.duration_window.append(duration)
        
    def _display_episode(self, i, avg_score, avg_duration):
        # display updates
        print('\rEpisode {}\tAverage Score: {:.2f}\tAvg. Duration: {:.4f}s'.format(i, avg_score, avg_duration), end="")
        
        if i % self.print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tAvg. Duration: {:.4f}s'.format(i, avg_score, avg_duration))
            
    def _display_complete(self, n_episodes, duration, avg_score):                
        print(
            '\rTraining complete, Average Score: {:.2f}\tTotal Time: {:.2f}m\tAvg. Episode Duration: {:.3f}s'.format(
            avg_score, duration/60.0, duration/n_episodes))

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
