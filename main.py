from environment import *
from agent import *
from game import *

import cvxpy as cp

#create the game instance
game_1=grid_game()

#create agents
agent_1=Agent_ps(game_1,10,False)
agent_2=Agent_fp_true(game_1,10,True)

#execute the game
play_eq(agent_1, agent_2, game_1, tau=10, episodes=2000)
