import numpy as np
import cvxpy as cp
#compute |V_max-v_min|
def compute_value_diff(V_max, V_min,game,init_state=-1):
    state_num=game.state_num
    init_prob=np.ones(state_num)/state_num
    # Calculate the absolute difference between the two value functions
    if init_state==-1:
      value_diff = np.dot((V_max[0] - V_min[0]),init_prob)
    else:
      value_diff = V_max[0,init_state] - V_min[0,init_state]

    return value_diff

def solve_value_max(game,tau):
      T=game.T
      R=game.R
      state_num=game.state_num
      action_1_num=game.action_1_num
      action_2_num=game.action_2_num
      V = np.zeros((tau+1,state_num))
      for t in range(tau):
            for s in range(state_num):
                # Initialize matrix Q(s) for mixed strategy evaluation
                Q_s = np.zeros((action_1_num, action_2_num))

                # Populate Q(s) with R(a1, a2) + T(s, a1, a2) * V for each action pair
                for a1 in range(action_1_num):
                    for a2 in range(action_2_num):
                        Q_s[a1, a2] = R[s, a1, a2] + np.dot(T[s, a1, a2], V[tau - t])

                # Define variables for mixed strategies
                u = cp.Variable(1)
                pi1 = cp.Variable(action_1_num)
                pi2 = cp.Variable(action_2_num)

                # Constraints for mixed strategies (non-negative and sum to 1)

                Q_s = cp.Constant(Q_s)


                constraints_max = [
                    cp.multiply(u , cp.Constant(np.ones(action_2_num))) - pi1@ Q_s <= np.zeros(action_2_num),
                    pi1 >= np.zeros(action_1_num),
                    cp.sum(pi1) == 1
                ]

                constraints_min = [
                    cp.multiply(u,cp.Constant(np.ones(action_1_num)))- Q_s@pi2 >=np.zeros(action_1_num),
                    pi2 >= np.zeros(action_2_num), cp.sum(pi2) == 1
                ]

                # Objective: maximize for player 1 and minimize for player 2

                # Maximizing player
                objective = cp.Maximize(u)
                prob = cp.Problem(objective, constraints_max)

                # Solve optimization
                prob.solve()

                # Store the optimal value for this timestep and state
                V[tau - t - 1, s] = prob.value
      return V

def solve_value_min(game,tau):
      T=game.T
      R=game.R
      state_num=game.state_num
      action_1_num=game.action_1_num
      action_2_num=game.action_2_num
      V = np.zeros((tau+1,state_num))
      for t in range(tau):
            for s in range(state_num):
                # Initialize matrix Q(s) for mixed strategy evaluation
                Q_s = np.zeros((action_1_num, action_2_num))

                # Populate Q(s) with R(a1, a2) + T(s, a1, a2) * V for each action pair
                for a1 in range(action_1_num):
                    for a2 in range(action_2_num):
                        Q_s[a1, a2] = R[s, a1, a2] + np.dot(T[s, a1, a2], V[tau - t])

                # Define variables for mixed strategies
                u = cp.Variable(1)
                pi1 = cp.Variable(action_1_num)
                pi2 = cp.Variable(action_2_num)

                # Constraints for mixed strategies (non-negative and sum to 1)

                Q_s = cp.Constant(Q_s)


                constraints_max = [
                    cp.multiply(u , cp.Constant(np.ones(action_2_num))) - pi1@ Q_s <= np.zeros(action_2_num),
                    pi1 >= np.zeros(action_1_num),
                    cp.sum(pi1) == 1
                ]

                constraints_min = [
                    cp.multiply(u,cp.Constant(np.ones(action_1_num)))- Q_s@pi2 >=np.zeros(action_1_num),
                    pi2 >= np.zeros(action_2_num), cp.sum(pi2) == 1
                ]

                # Objective: maximize for player 1 and minimize for player 2
                objective = cp.Minimize(u)
                prob = cp.Problem(objective, constraints_min)

                # Solve optimization
                prob.solve()

                # Store the optimal value for this timestep and state
                V[tau - t - 1, s] = prob.value
      return V


def solve_value_by_strategy(s,game,tau,pi):
    T=game.T
    R=game.R
    state_num=game.state_num
    action_1_num=game.action_1_num
    V = np.zeros((tau+1,state_num))
    for t in range(tau):  # Run value iteration
      for s in range(state_num):
          V[tau-t-1,s] = max(np.dot(R[s, a1] + np.dot(T[s, a1], V[tau-t]),pi[t][s]) for a1 in range(action_1_num))
    return V[0,s]
