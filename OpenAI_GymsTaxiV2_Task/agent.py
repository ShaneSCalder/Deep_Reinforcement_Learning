import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1 

    def select_action(self, state):
        
        self.epsilon = 1
        self.decay = 0.5
        self.gamma = 0.9
        self.alpha = 0.3
        
        
        
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
       
        
        
       
        
         
         

    def step(self, state, action, reward, next_state, done, alpha=0.1, gamma=1.0):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += alpha * (reward + (gamma * np.max(self.Q[next_state])) - self.Q[state][action])