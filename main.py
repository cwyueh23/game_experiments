from environment import *
from agent import *
from game import *
import test

import cvxpy as cp

#create the game instance
game_1=grid_game()

#create agents
agent_01=Agent_ps(game_1,10,False,seed=1915)
agent_02=Agent_ps(game_1,10,False,seed=1915)
agent_03=Agent_ps(game_1,10,False,seed=1915)
agent_04=Agent_ps(game_1,10,False,seed=1915)
agent_05=Agent_ps(game_1,10,False,seed=1915)
agent_06=Agent_ps(game_1,10,False,seed=1915)
agent_07=Agent_ps(game_1,10,False,seed=1915)
agent_08=Agent_ps(game_1,10,False,seed=1915)
agent_09=Agent_ps(game_1,10,False,seed=1915)

agent_1=Agent_eq(game_1,10,True,seed=45)
agent_2=Agent_ps_map(game_1,10,True,seed=7736)
agent_3=Agent_ps(game_1,10,True,seed=82714)
agent_4=Agent_fp_true(game_1,10,True,seed=2571)
agent_5=Agent_fp_map(game_1,10,True,seed=4663)
agent_6=Agent_fp(game_1,10,True,seed=284)
agent_7=Agent_sample_true(game_1,10,True,seed=681)
agent_8=Agent_sample_map(game_1,10,True,seed=9892)
agent_9=Agent_sample_ps(game_1,10,True,seed=2)

agent_test=test.Agent_fp_best(game_1,10,True,seed=9140)


#execute the game
play_eq( agent_01,agent_1, game_1, tau=10, episodes=1000,filename='1')
play_eq( agent_02,agent_2, game_1, tau=10, episodes=1000,filename='2')
play_eq( agent_03,agent_3, game_1, tau=10, episodes=1000,filename='3')
play_eq( agent_04,agent_4, game_1, tau=10, episodes=1000,filename='4')
play_eq( agent_05,agent_5, game_1, tau=10, episodes=1000,filename='5')
play_eq( agent_06,agent_6, game_1, tau=10, episodes=1000,filename='6')
play_eq( agent_07,agent_7, game_1, tau=10, episodes=1000,filename='7')
play_eq( agent_08,agent_8, game_1, tau=10, episodes=1000,filename='8')
play_eq( agent_09,agent_9, game_1, tau=10, episodes=1000,filename='9')

'''
a,b=play_single( agent_0,agent_test, game_1, tau=10, episodes=1000,filename='best_reverse',test_best=1)
c,d=play_single( agent_00,agent_1, game_1, tau=10, episodes=1000,filename='ps_reverse')
e,f=play_single( agent_000,agent_4, game_1, tau=10, episodes=1000,filename='fp_reverse')
g,h=play_single( agent_0000,agent_7, game_1, tau=10, episodes=1000,filename='sample_reverse')

for i in range(1000):
    b[i]=b[i]*(i+1)
    d[i]=d[i]*(i+1)
    f[i]=f[i]*(i+1)
    h[i]=h[i]*(i+1)
    
plt.plot(range(1000),b, label='Best FP')
plt.plot(range(1000),d, label='PS')
plt.plot(range(1000),f, label='FP')
plt.plot(range(1000),h, label='Sample')
plt.legend()    
plt.xlabel('Episodes')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret Comparison')
plt.savefig('bestcumulative_regret_reverse.png')
'''