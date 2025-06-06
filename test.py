import numpy as np
import cvxpy as cp
import utility
from agent import Agent

class Agent_fp_best(Agent):
    def __init__(self, game, horizon, is_minimizer=False,larger_prior=False,seed=292):
        super().__init__(game, horizon, is_minimizer,seed)
        if is_minimizer:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_1_num))/self.action_1_num for i in range(horizon) ]
            #self.V=utility.solve_value_by_strategy(self.T,game, horizon,maximal=False)
            #self.opponent_history=[ np.ones((self.state_num,self.action_1_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_2_num))
        else:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_2_num))/self.action_2_num for i in range(horizon) ]
            #self.V=utility.solve_value_by_strategy(game, horizon,maximal=True)
            #self.opponent_history=[ np.ones((self.state_num,self.action_2_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_1_num))
        self.R=game.R
        self.T = game.T
        self.V=np.zeros((self.horizon+1,self.state_num))

    def solve_value(self,opponent_strategy=None):
        if opponent_strategy is not None:
            self.opponent_strategy = opponent_strategy
        else:
            print("Warning: No opponent strategy provided, using uniform strategy.")
        
        for t in range(self.horizon):  # Run value iteration
            for s in range(self.state_num):
                if self.is_minimizer:
                    self.V[self.horizon-t-1,s] = min(np.dot(self.R[s, :,a2] + np.dot(self.T[s,:, a2], self.V[self.horizon-t]),opponent_strategy[self.horizon-t-1][s]) for a2 in range(self.action_2_num))
                    action=np.argmin([np.dot(self.R[s, :,a2] + np.dot(self.T[s,:, a2], self.V[self.horizon-t]),opponent_strategy[self.horizon-t-1][s]) for a2 in range(self.action_2_num)])
                    self.strategy[self.horizon-t-1,s]=np.zeros(self.action_2_num)
                    self.strategy[self.horizon-t-1,s,action]=1
                else:
                    self.V[self.horizon-t-1,s] = max(np.dot(self.R[s, a1] + np.dot(self.T[s, a1], self.V[self.horizon-t]),opponent_strategy[self.horizon-t-1][s]) for a1 in range(self.action_1_num))
                    action=np.argmax([np.dot(self.R[s, a1] + np.dot(self.T[s, a1], self.V[self.horizon-t]),opponent_strategy[self.horizon-t-1][s]) for a1 in range(self.action_1_num)])
                    self.strategy[self.horizon-t-1,s]=np.zeros(self.action_1_num)
                    self.strategy[self.horizon-t-1,s,action]=1
    
    def choose_action(self, s, t):
        '''
        if np.sum(self.strategy[t,s])==0:
            print("Warning: strategy is zero",t,s)
        '''
        #self.strategy[t,s] /= np.sum(self.strategy[t,s])
        if not self.is_minimizer:
            action = np.random.choice(range(self.action_1_num), p=self.strategy[t,s])
        else:
            action = np.random.choice(range(self.action_2_num), p=self.strategy[t,s])
        return action


    def value(self,s):
      return self.V[0,s]

    def get_strategy(self):
      return self.strategy
  
class Agent_arbitrary(Agent):
    def __init__(self, game, horizon, is_minimizer=False,larger_prior=False,seed=292,strategy=None):
        super().__init__(game, horizon, is_minimizer,seed)
        if is_minimizer:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_1_num))/self.action_1_num for i in range(horizon) ]
            #self.V=utility.solve_value_by_strategy(self.T,game, horizon,maximal=False)
            self.opponent_history=[ np.ones((self.state_num,self.action_1_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_2_num))
            if strategy is not None:
                self.strategy = strategy
            else:
                self.strategy = np.ones((horizon, self.state_num, self.action_2_num)) / self.action_2_num
        else:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_2_num))/self.action_2_num for i in range(horizon) ]
            #self.V=utility.solve_value_by_strategy(game, horizon,maximal=True)
            self.opponent_history=[ np.ones((self.state_num,self.action_2_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_1_num))
            if strategy is not None:
                self.strategy = strategy
            else:
                self.strategy = np.ones((horizon, self.state_num, self.action_1_num)) / self.action_1_num
        self.R=game.R
        self.T = game.T
        self.V=np.zeros((self.horizon+1,self.state_num))
        
        
    def choose_action(self, s, t):
        if not self.is_minimizer:
            action = np.random.choice(range(self.action_1_num), p=self.strategy[t,s])
        else:
            action = np.random.choice(range(self.action_2_num), p=self.strategy[t,s])
        return action
    
    def value(self,s):
      return self.V[0,s]
  
    def get_strategy(self):
      return self.strategy
      