import numpy as np
import cvxpy as cp
import utility

class Agent:
    def __init__(self, game, horizon, is_minimizer=False,seed=292):
        self.state_num_1=game.state_num_1
        self.state_num_2=game.state_num_2
        self.state_num = game.state_num
        self.action_1_num = game.action_1_num
        self.action_2_num = game.action_2_num
        self.horizon = horizon
        self.is_minimizer=is_minimizer
        self.rng= np.random.default_rng(seed)  # Random number generator for reproducibility
        
    def sample_model(self):
        pass
    
    def choose_action(self, s, t):
        pass
    
    def get_strategy(self):
        pass

    def compute_measures(self, T):
        return 0,0,0
    
    def update_prior(self,next_s,s,a_1,a_2,tau):
        pass

#posterior sampling agent
    
class Agent_ps(Agent):
    def __init__(self, game, horizon, is_minimizer=False,larger_prior=False,seed=292):
        super().__init__(game, horizon, is_minimizer,seed)
        # Initialize prior as Dirichlet distributions (initially uniform) for transition probabilities
        if larger_prior:
            self.prior = np.ones((game.state_num_1*game.state_num_2, game.action_1_num*game.action_2_num, game.state_num_1*game.state_num_2))  # Dirichlet params
        else:
            self.prior_1 = np.ones((game.state_num_1, game.action_1_num, game.state_num_1))  # Dirichlet params
            self.prior_2 = np.ones((game.state_num_2, game.action_2_num, game.state_num_2))  # Dirichlet params
        self.larger_prior=larger_prior
        self.T = np.zeros((self.state_num, game.action_1_num, game.action_2_num, self.state_num))  # Transition probabilities (initialized to zero)
        self.V=np.zeros((self.horizon+1,self.state_num))
        self.R=game.R
        #strategy dimension: horizon x state_num x action_num
        #self.strategy=[[] for i in range(horizon) ]
        if is_minimizer:
            self.strategy=np.zeros((horizon,self.state_num,self.action_2_num))
        else:   
            self.strategy=np.zeros((horizon,self.state_num,self.action_1_num))


    def update_T(self,T_1,T_2):
        for s in range(self.state_num):
          for a_1 in range(self.action_1_num):
              for a_2 in range(self.action_2_num):
                for s_next in range(self.state_num):
                  self.T[s,a_1,a_2,s_next] = T_1[s%9,a_1,s_next%9]*T_2[s//9,a_2,s_next//9]


    def sample_model(self):
      if self.larger_prior:
          for s in range(self.state_num):
            for a1 in range(self.action_1_num):
                for a2 in range(self.action_2_num):
                    self.T[s, a1,a2] = self.rng.dirichlet(self.prior[s, a1,a2])

      else:
        T_1=np.zeros((self.state_num_1,self.action_1_num,self.state_num_1))
        T_2=np.zeros((self.state_num_2,self.action_2_num,self.state_num_2))
        # Sample transition probabilities from the Dirichlet prior for each state-action pair
        for s in range(self.state_num_1):
            for a1 in range(self.action_1_num):
                T_1[s, a1] = self.rng.dirichlet(self.prior_1[s, a1])
        for s in range(self.state_num_2):
            for a2 in range(self.action_2_num):
                T_2[s, a2] = self.rng.dirichlet(self.prior_2[s, a2])
        self.update_T(T_1,T_2)

    def update_prior(self,next_s,s,a_1,a_2,tau):
        if self.larger_prior:
            self.prior[s,a_1,a_2,next_s]+=1
        else:
            self.prior_1[s%9,a_1,next_s%9]+=1
            self.prior_2[s//9,a_2,next_s//9]+=1


    def solve_value(self):
        #self.strategy=[[] for _ in range(self.horizon)]
        for t in range(self.horizon):
            for s in range(self.state_num):
                # Initialize matrix Q(s) for mixed strategy evaluation
                Q_s = np.zeros((self.action_1_num, self.action_2_num))

                # Populate Q(s) with R(a1, a2) + T(s, a1, a2) * V for each action pair
                for a1 in range(self.action_1_num):
                    for a2 in range(self.action_2_num):
                        Q_s[a1, a2] = self.R[s, a1, a2] + np.dot(self.T[s, a1, a2], self.V[self.horizon - t])

                # Define variables for mixed strategies
                u = cp.Variable(1)
                pi1 = cp.Variable(self.action_1_num)
                pi2 = cp.Variable(self.action_2_num)

                # Constraints for mixed strategies (non-negative and sum to 1)

                Q_s = cp.Constant(Q_s)


                constraints_max = [
                    cp.multiply(u , cp.Constant(np.ones(self.action_2_num))) - pi1@ Q_s <= np.zeros(self.action_2_num),
                    pi1 >= np.zeros(self.action_1_num),
                    cp.sum(pi1) == 1
                ]

                constraints_min = [
                    cp.multiply(u,cp.Constant(np.ones(self.action_1_num)))- Q_s@pi2 >=np.zeros(self.action_1_num),
                    pi2 >= np.zeros(self.action_2_num), cp.sum(pi2) == 1
                ]

                # Objective: maximize for player 1 and minimize for player 2
                if not self.is_minimizer:
                    # Maximizing player
                    objective = cp.Maximize(u)
                    prob = cp.Problem(objective, constraints_max)
                    prob.solve()
                    pi1_opt = pi1.value
                    pi1_opt = np.maximum(pi1_opt, 0)  # Ensure all values are non-negative
                    pi1_opt /= pi1_opt.sum()
                    self.strategy[t-1,s]=pi1_opt
                else:
                    # Minimizing player
                    objective = cp.Minimize(u)
                    prob = cp.Problem(objective, constraints_min)
                    prob.solve()
                    pi2_opt = pi2.value
                    pi2_opt = np.maximum(pi2_opt, 0)  # Ensure all values are non-negative
                    pi2_opt /= pi2_opt.sum()
                    self.strategy[t-1,s]=pi2_opt


                # Store the optimal value for this timestep and state
                self.V[self.horizon - t - 1, s] = prob.value



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

    def compute_measures(self, T):
        """
        Compute various distance measures between two tensors self_T and T.

        Parameters:
            self_T (np.ndarray): The first tensor, shape (81, 4, 4, 81).
            T (np.ndarray): The second tensor, shape (81, 4, 4, 81).

        Returns:
            dict: A dictionary containing the computed measures.
        """
        self_T = self.T
        # Ensure tensors have the correct dimensions
        assert self_T.shape == T.shape == (self.state_num, self.action_1_num, self.action_2_num, self.state_num), "Tensors must have shape (81, 4, 4, 81)."

        # Compute element-wise difference
        diff = self_T - T

        # Reshape the difference tensor into a matrix of shape (81*4*4, 81)
        diff_matrix = diff.reshape(self.state_num * self.action_1_num * self.action_2_num, self.state_num)

        # 1. Frobenius Norm
        frobenius_norm = np.sqrt(np.sum(diff_matrix**2))

        # 2. L1 Norm (maximum column-wise deviation per reshaped matrix)
        l1_norm = np.max(np.sum(np.abs(diff_matrix), axis=0))

        # 3. L∞ Norm (maximum row-wise deviation per reshaped matrix)
        linf_norm = np.max(np.sum(np.abs(diff_matrix), axis=1))

        return frobenius_norm, l1_norm, linf_norm



class Agent_eq(Agent):
    def __init__(self, game, horizon, is_minimizer=False,seed=292):
        # Initialize prior as Dirichlet distributions (initially uniform) for transition probabilities
        super().__init__(game, horizon, is_minimizer)
        self.prior = np.ones((self.state_num, game.action_1_num, game.action_2_num, self.state_num))  # Dirichlet params
        self.T = game.T  # Transition probabilities known
        if is_minimizer:
            self.V=utility.solve_value(game.T,game, horizon,maximum=False)
        else:
            self.V=utility.solve_value(game.T,game, horizon,maximum=True)
        self.R=game.R
        self.strategy=[[] for i in range(horizon) ]


    def solve_value(self):
        for t in range(self.horizon):
            for s in range(self.state_num):
                # Initialize matrix Q(s) for mixed strategy evaluation
                Q_s = np.zeros((self.action_1_num, self.action_2_num))

                # Populate Q(s) with R(a1, a2) + T(s, a1, a2) * V for each action pair
                for a1 in range(self.action_1_num):
                    for a2 in range(self.action_2_num):
                        Q_s[a1, a2] = self.R[s, a1, a2] + np.dot(self.T[s, a1, a2], self.V[self.horizon - t])

                # Define variables for mixed strategies
                u = cp.Variable(1)
                pi1 = cp.Variable(self.action_1_num)
                pi2 = cp.Variable(self.action_2_num)

                # Constraints for mixed strategies (non-negative and sum to 1)

                Q_s = cp.Constant(Q_s)


                constraints_max = [
                    cp.multiply(u , cp.Constant(np.ones(self.action_2_num))) - pi1@ Q_s <= np.zeros(self.action_2_num),
                    pi1 >= np.zeros(self.action_1_num),
                    cp.sum(pi1) == 1
                ]

                constraints_min = [
                    cp.multiply(u,cp.Constant(np.ones(self.action_1_num)))- Q_s@pi2 >=np.zeros(self.action_1_num),
                    pi2 >= np.zeros(self.action_2_num), cp.sum(pi2) == 1
                ]

                # Objective: maximize for player 1 and minimize for player 2
                if not self.is_minimizer:
                    # Maximizing player
                    objective = cp.Maximize(u)
                    prob = cp.Problem(objective, constraints_max)
                    prob.solve()
                    pi1_opt = pi1.value
                    pi1_opt = np.maximum(pi1_opt, 0)  # Ensure all values are non-negative
                    pi1_opt /= pi1_opt.sum()
                    self.strategy[t-1].append(pi1_opt)
                else:
                    # Minimizing player
                    objective = cp.Minimize(u)
                    prob = cp.Problem(objective, constraints_min)
                    prob.solve()
                    pi2_opt = pi2.value
                    pi2_opt = np.maximum(pi2_opt, 0)  # Ensure all values are non-negative
                    pi2_opt /= pi2_opt.sum()
                    self.strategy[t-1].append(pi2_opt)

                # Solve optimization

                # Store the optimal value for this timestep and state
                self.V[self.horizon - t - 1, s] = prob.value

                # Store the optimal value for this timestep and state



    def choose_action(self, s, t):
        # Initialize the matrix Q(s) as done in solve_value to compute mixed strategies
        Q_s = np.zeros((self.action_1_num, self.action_2_num))
        for a1 in range(self.action_1_num):
            for a2 in range(self.action_2_num):
                Q_s[a1, a2] = self.R[s, a1, a2] + np.dot(self.T[s, a1, a2], self.V[t + 1])

        # Define variables for mixed strategies
        u = cp.Variable(1)
        pi1 = cp.Variable(self.action_1_num)
        pi2 = cp.Variable(self.action_2_num)

        # Set constraints for probabilities (non-negative and sum to 1)
        constraints_max = [
                    cp.multiply(u , cp.Constant(np.ones(self.action_2_num))) - pi1@ Q_s <= np.zeros(self.action_2_num),
                    pi1 >= np.zeros(self.action_1_num),
                    cp.sum(pi1) == 1
                ]

        constraints_min = [
                    cp.multiply(u,cp.Constant(np.ones(self.action_1_num)))- Q_s@pi2 >=np.zeros(self.action_1_num),
                    pi2 >= np.zeros(self.action_2_num), cp.sum(pi2) == 1
                ]

        # Objective: maximizing for player 1 and minimizing for player 2
        if not self.is_minimizer:
            # Player 1 (maximizer) strategy
            objective = cp.Maximize(u)
            prob = cp.Problem(objective, constraints_max)
            prob.solve()
            pi1_opt = pi1.value
            pi1_opt = np.maximum(pi1_opt, 0)  # Ensure all values are non-negative
            pi1_opt /= pi1_opt.sum()  # Normalize to sum to 1

            # Sample action for Player 1 based on mixed strategy
            action = np.random.choice(range(self.action_1_num), p=pi1_opt)
        else:
            # Player 2 (minimizer) strategy
            objective = cp.Minimize(u)
            prob = cp.Problem(objective, constraints_min)
            prob.solve()
            pi2_opt = pi2.value
            pi2_opt = np.maximum(pi2_opt, 0)  # Ensure all values are non-negative
            pi2_opt /= pi2_opt.sum()  # Normalize to sum to 1
            # Sample action for Player 2 based on mixed strategy
            action = np.random.choice(range(self.action_2_num), p=pi2_opt)

        return action


    def value(self,s):
      return self.V[0,s]

    def get_strategy(self):
      return self.strategy

class Agent_ps_map(Agent):
    def __init__(self, game, horizon, is_minimizer=False,larger_prior=False,seed=292):
        super().__init__(game, horizon, is_minimizer,seed)
        # Initialize prior as Dirichlet distributions (initially uniform) for transition probabilities
        if larger_prior:
            self.prior = 1.1*np.ones((game.state_num_1*game.state_num_2, game.action_1_num*game.action_2_num, game.state_num_1*game.state_num_2))  # Dirichlet params
        else:
            self.prior_1 = 1.1*np.ones((game.state_num_1, game.action_1_num, game.state_num_1))  # Dirichlet params
            self.prior_2 = 1.1*np.ones((game.state_num_2, game.action_2_num, game.state_num_2))  # Dirichlet params
        self.larger_prior=larger_prior
        self.T = np.zeros((self.state_num, game.action_1_num, game.action_2_num, self.state_num))  # Transition probabilities (initialized to zero)
        self.V=np.zeros((self.horizon+1,self.state_num))
        self.R=game.R
        self.strategy=[[] for i in range(horizon) ]
        

    def update_T(self,T_1,T_2):
        for s in range(self.state_num):
          for a_1 in range(self.action_1_num):
              for a_2 in range(self.action_2_num):
                for s_next in range(self.state_num):
                  self.T[s,a_1,a_2,s_next] = T_1[s%9,a_1,s_next%9]*T_2[s//9,a_2,s_next//9]


    def sample_model(self):
      if self.larger_prior:
          for s in range(self.state_num):
            for a1 in range(self.action_1_num):
                for a2 in range(self.action_2_num):
                    for s_next in range(self.state_num):
                        self.T[s, a1,a2,s_next] = (self.prior[s, a1,a2,s_next]-1)/(np.sum(self.prior[s, a1,a2])-self.state_num)

      else:
        T_1=np.zeros((self.state_num_1,self.action_1_num,self.state_num_1))
        T_2=np.zeros((self.state_num_2,self.action_2_num,self.state_num_2))
        # Sample transition probabilities from the Dirichlet prior for each state-action pair
        for s in range(self.state_num_1):
            for a1 in range(self.action_1_num):
                for s_next in range(self.state_num_1):
                    T_1[s, a1,s_next] = (self.prior_1[s, a1,s_next]-1)/(np.sum(self.prior_1[s, a1])-self.state_num_1)
        for s in range(self.state_num_2):
            for a2 in range(self.action_2_num):
                for s_next in range(self.state_num_2):
                    T_2[s, a2,s_next] = (self.prior_2[s, a2,s_next]-1)/(np.sum(self.prior_2[s, a2])-self.state_num_2)
        self.update_T(T_1,T_2)

    def update_prior(self,next_s,s,a_1,a_2,tau):
        if self.larger_prior:
            self.prior[s,a_1,a_2,next_s]+=1
        else:
            self.prior_1[s%9,a_1,next_s%9]+=1
            self.prior_2[s//9,a_2,next_s//9]+=1


    def solve_value(self):
        self.strategy=[[] for _ in range(self.horizon)]
        for t in range(self.horizon):
            for s in range(self.state_num):
                # Initialize matrix Q(s) for mixed strategy evaluation
                Q_s = np.zeros((self.action_1_num, self.action_2_num))

                # Populate Q(s) with R(a1, a2) + T(s, a1, a2) * V for each action pair
                for a1 in range(self.action_1_num):
                    for a2 in range(self.action_2_num):
                        Q_s[a1, a2] = self.R[s, a1, a2] + np.dot(self.T[s, a1, a2], self.V[self.horizon - t])

                # Define variables for mixed strategies
                u = cp.Variable(1)
                pi1 = cp.Variable(self.action_1_num)
                pi2 = cp.Variable(self.action_2_num)

                # Constraints for mixed strategies (non-negative and sum to 1)

                Q_s = cp.Constant(Q_s)


                constraints_max = [
                    cp.multiply(u , cp.Constant(np.ones(self.action_2_num))) - pi1@ Q_s <= np.zeros(self.action_2_num),
                    pi1 >= np.zeros(self.action_1_num),
                    cp.sum(pi1) == 1
                ]

                constraints_min = [
                    cp.multiply(u,cp.Constant(np.ones(self.action_1_num)))- Q_s@pi2 >=np.zeros(self.action_1_num),
                    pi2 >= np.zeros(self.action_2_num), cp.sum(pi2) == 1
                ]

                # Objective: maximize for player 1 and minimize for player 2
                if not self.is_minimizer:
                    # Maximizing player
                    objective = cp.Maximize(u)
                    prob = cp.Problem(objective, constraints_max)
                    prob.solve()
                    pi1_opt = pi1.value
                    pi1_opt = np.maximum(pi1_opt, 0)  # Ensure all values are non-negative
                    pi1_opt /= pi1_opt.sum()
                    self.strategy[t-1].append(pi1_opt)
                else:
                    # Minimizing player
                    objective = cp.Minimize(u)
                    prob = cp.Problem(objective, constraints_min)
                    prob.solve()
                    pi2_opt = pi2.value
                    pi2_opt = np.maximum(pi2_opt, 0)  # Ensure all values are non-negative
                    pi2_opt /= pi2_opt.sum()
                    self.strategy[t-1].append(pi2_opt)


                # Store the optimal value for this timestep and state
                self.V[self.horizon - t - 1, s] = prob.value



    def choose_action(self, s, t):
        if not self.is_minimizer:
            action = np.random.choice(range(self.action_1_num), p=self.strategy[t][s])
        else:
            action = np.random.choice(range(self.action_2_num), p=self.strategy[t][s])
        return action


    def value(self,s):
      return self.V[0,s]

    def get_strategy(self):
      return self.strategy

    def compute_measures(self, T):
        """
        Compute various distance measures between two tensors self_T and T.

        Parameters:
            self_T (np.ndarray): The first tensor, shape (81, 4, 4, 81).
            T (np.ndarray): The second tensor, shape (81, 4, 4, 81).

        Returns:
            dict: A dictionary containing the computed measures.
        """
        self_T = self.T
        # Ensure tensors have the correct dimensions
        assert self_T.shape == T.shape == (self.state_num, self.action_1_num, self.action_2_num, self.state_num), "Tensors must have shape (81, 4, 4, 81)."

        # Compute element-wise difference
        diff = self_T - T

        # Reshape the difference tensor into a matrix of shape (81*4*4, 81)
        diff_matrix = diff.reshape(self.state_num * self.action_1_num * self.action_2_num, self.state_num)

        # 1. Frobenius Norm
        frobenius_norm = np.sqrt(np.sum(diff_matrix**2))

        # 2. L1 Norm (maximum column-wise deviation per reshaped matrix)
        l1_norm = np.max(np.sum(np.abs(diff_matrix), axis=0))

        # 3. L∞ Norm (maximum row-wise deviation per reshaped matrix)
        linf_norm = np.max(np.sum(np.abs(diff_matrix), axis=1))

        return frobenius_norm, l1_norm, linf_norm

class Agent_fp(Agent):
    def __init__(self, game, horizon, is_minimizer=False,larger_prior=False,seed=292):
        super().__init__(game, horizon, is_minimizer,seed)
        if is_minimizer:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_1_num))/self.action_1_num for i in range(horizon) ]
            #self.V=utility.solve_value_by_strategy(game, horizon,maximal=False)
            self.opponent_history=[ np.ones((self.state_num,self.action_1_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_2_num))
        else:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_2_num))/self.action_2_num for i in range(horizon) ]
            #self.V=utility.solve_value_by_strategy(game, horizon,maximal=True)
            self.opponent_history=[ np.ones((self.state_num,self.action_2_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_1_num))
        self.R=game.R
        self.T = np.zeros((self.state_num, game.action_1_num, game.action_2_num, self.state_num))  # Transition probabilities (initialized to zero)
        self.V=np.zeros((self.horizon+1,self.state_num))
        self.larger_prior=larger_prior
        if larger_prior:
            self.prior = np.ones((game.state_num_1*game.state_num_2, game.action_1_num*game.action_2_num, game.state_num_1*game.state_num_2))  # Dirichlet params
        else:
            self.prior_1 = np.ones((game.state_num_1, game.action_1_num, game.state_num_1))  # Dirichlet params
            self.prior_2 = np.ones((game.state_num_2, game.action_2_num, game.state_num_2))  # Dirichlet params
        
    def update_T(self,T_1,T_2):
        for s in range(self.state_num):
          for a_1 in range(self.action_1_num):
              for a_2 in range(self.action_2_num):
                for s_next in range(self.state_num):
                  self.T[s,a_1,a_2,s_next] = T_1[s%9,a_1,s_next%9]*T_2[s//9,a_2,s_next//9]


    def sample_model(self):
        if self.larger_prior:
          for s in range(self.state_num):
            for a1 in range(self.action_1_num):
                for a2 in range(self.action_2_num):
                    self.T[s, a1,a2] = self.rng.dirichlet(self.prior[s, a1,a2])

        else:
            T_1=np.zeros((self.state_num_1,self.action_1_num,self.state_num_1))
            T_2=np.zeros((self.state_num_2,self.action_2_num,self.state_num_2))
            # Sample transition probabilities from the Dirichlet prior for each state-action pair
            for s in range(self.state_num_1):
                for a1 in range(self.action_1_num):
                    for s_next in range(self.state_num_1):
                        T_1[s, a1,s_next] = (self.prior_1[s, a1,s_next]-1)/(np.sum(self.prior_1[s, a1])-self.state_num_1)
            for s in range(self.state_num_2):
                for a2 in range(self.action_2_num):
                    for s_next in range(self.state_num_2):
                        T_2[s, a2,s_next] = (self.prior_2[s, a2,s_next]-1)/(np.sum(self.prior_2[s, a2])-self.state_num_2)
            self.update_T(T_1,T_2)

    def update_prior(self,next_s,s,a_1,a_2,tau):
        if self.larger_prior:
            self.prior[s,a_1,a_2,next_s]+=1
        else:
            self.prior_1[s%9,a_1,next_s%9]+=1
            self.prior_2[s//9,a_2,next_s//9]+=1
        # Update opponent's history and strategy
        if self.is_minimizer:
            self.opponent_history[tau][s,a_1]+=1
            for state in range(self.state_num):
                for a in range(self.action_1_num):
                    self.opponent_strategy[tau][state,a] = (self.opponent_history[tau][state,a]) / np.sum(self.opponent_history[tau][state])
        else:
            self.opponent_history[tau][s,a_2]+=1
            for state in range(self.state_num):
                for a in range(self.action_2_num):
                    self.opponent_strategy[tau][state,a] = (self.opponent_history[tau][state,a]) / np.sum(self.opponent_history[tau][state])

    def solve_value(self):
        for t in range(self.horizon):  # Run value iteration
            for s in range(self.state_num):
                if self.is_minimizer:
                    self.V[self.horizon-t-1,s] = min(np.dot(self.R[s, :,a2] + np.dot(self.T[s,:, a2], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a2 in range(self.action_2_num))
                    action=np.argmin(np.dot(self.R[s, :,a2] + np.dot(self.T[s,:, a2], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a2 in range(self.action_2_num))
                    self.strategy[t]=np.zeros((self.state_num,self.action_2_num))
                    self.strategy[t][s,action]=1
                else:
                    self.V[self.horizon-t-1,s] = max(np.dot(self.R[s, a1] + np.dot(self.T[s, a1], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a1 in range(self.action_1_num))
                    action=np.argmax(np.dot(self.R[s, a1] + np.dot(self.T[s, a1], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a1 in range(self.action_1_num))
                    self.strategy[t]=np.zeros((self.state_num,self.action_1_num))
                    self.strategy[t][s,action]=1

    
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

    def compute_measures(self, T):
        """
        Compute various distance measures between two tensors self_T and T.

        Parameters:
            self_T (np.ndarray): The first tensor, shape (81, 4, 4, 81).
            T (np.ndarray): The second tensor, shape (81, 4, 4, 81).

        Returns:
            dict: A dictionary containing the computed measures.
        """
        self_T = self.T
        # Ensure tensors have the correct dimensions
        assert self_T.shape == T.shape == (self.state_num, self.action_1_num, self.action_2_num, self.state_num), "Tensors must have shape (81, 4, 4, 81)."

        # Compute element-wise difference
        diff = self_T - T

        # Reshape the difference tensor into a matrix of shape (81*4*4, 81)
        diff_matrix = diff.reshape(self.state_num * self.action_1_num * self.action_2_num, self.state_num)

        # 1. Frobenius Norm
        frobenius_norm = np.sqrt(np.sum(diff_matrix**2))

        # 2. L1 Norm (maximum column-wise deviation per reshaped matrix)
        l1_norm = np.max(np.sum(np.abs(diff_matrix), axis=0))

        # 3. L∞ Norm (maximum row-wise deviation per reshaped matrix)
        linf_norm = np.max(np.sum(np.abs(diff_matrix), axis=1))

        return frobenius_norm, l1_norm, linf_norm



class Agent_fp_map(Agent):
    def __init__(self, game, horizon, is_minimizer=False,larger_prior=False,seed=292):
        super().__init__(game, horizon, is_minimizer,seed)
        if is_minimizer:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_1_num))/self.action_1_num for i in range(horizon) ]
            #self.V=utility.solve_value_by_strategy(self.T,game, horizon,maximal=False)
            self.opponent_history=[ np.ones((self.state_num,self.action_1_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_2_num))
        else:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_2_num))/self.action_2_num for i in range(horizon) ]
            #self.V=utility.solve_value_by_strategy(game, horizon,maximal=True)
            self.opponent_history=[ np.ones((self.state_num,self.action_2_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_1_num))
        self.R=game.R
        self.T = np.zeros((self.state_num, game.action_1_num, game.action_2_num, self.state_num))  # Transition probabilities (initialized to zero)
        self.V=np.zeros((self.horizon+1,self.state_num))
        self.larger_prior=larger_prior
        if larger_prior:
            self.prior = np.ones((game.state_num_1*game.state_num_2, game.action_1_num*game.action_2_num, game.state_num_1*game.state_num_2))  # Dirichlet params
        else:
            self.prior_1 = np.ones((game.state_num_1, game.action_1_num, game.state_num_1))  # Dirichlet params
            self.prior_2 = np.ones((game.state_num_2, game.action_2_num, game.state_num_2))  # Dirichlet params
        
    def update_T(self,T_1,T_2):
        for s in range(self.state_num):
          for a_1 in range(self.action_1_num):
              for a_2 in range(self.action_2_num):
                for s_next in range(self.state_num):
                  self.T[s,a_1,a_2,s_next] = T_1[s%9,a_1,s_next%9]*T_2[s//9,a_2,s_next//9]


    def sample_model(self):
        if self.larger_prior:
          for s in range(self.state_num):
            for a1 in range(self.action_1_num):
                for a2 in range(self.action_2_num):
                    for s_next in range(self.state_num):
                        self.T[s, a1,a2,s_next] = (self.prior[s, a1,a2,s_next]-1)/(np.sum(self.prior[s, a1,a2])-self.state_num)

        else:
            T_1=np.zeros((self.state_num_1,self.action_1_num,self.state_num_1))
            T_2=np.zeros((self.state_num_2,self.action_2_num,self.state_num_2))
            # Sample transition probabilities from the Dirichlet prior for each state-action pair
            for s in range(self.state_num_1):
                for a1 in range(self.action_1_num):
                    for s_next in range(self.state_num_1):
                        T_1[s, a1,s_next] = (self.prior_1[s, a1,s_next]-1)/(np.sum(self.prior_1[s, a1])-self.state_num_1)
            for s in range(self.state_num_2):
                for a2 in range(self.action_2_num):
                    for s_next in range(self.state_num_2):
                        T_2[s, a2,s_next] = (self.prior_2[s, a2,s_next]-1)/(np.sum(self.prior_2[s, a2])-self.state_num_2)
            self.update_T(T_1,T_2)


    def update_prior(self,next_s,s,a_1,a_2,tau):
        if self.larger_prior:
            self.prior[s,a_1,a_2,next_s]+=1
        else:
            self.prior_1[s%9,a_1,next_s%9]+=1
            self.prior_2[s//9,a_2,next_s//9]+=1
        # Update opponent's history and strategy
        if self.is_minimizer:
            self.opponent_history[tau][s,a_1]+=1
            for state in range(self.state_num):
                for a in range(self.action_1_num):
                    self.opponent_strategy[tau][state,a] = (self.opponent_history[tau][state,a]) / np.sum(self.opponent_history[tau][state])
        else:
            self.opponent_history[tau][s,a_2]+=1
            for state in range(self.state_num):
                for a in range(self.action_2_num):
                    self.opponent_strategy[tau][state,a] = (self.opponent_history[tau][state,a]) / np.sum(self.opponent_history[tau][state])

    def solve_value(self):
        for t in range(self.horizon):  # Run value iteration
            for s in range(self.state_num):
                if self.is_minimizer:
                    self.V[self.horizon-t-1,s] = min(np.dot(self.R[s, :,a2] + np.dot(self.T[s,:, a2], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a2 in range(self.action_2_num))
                    action=np.argmin(np.dot(self.R[s, :,a2] + np.dot(self.T[s,:, a2], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a2 in range(self.action_2_num))
                    self.strategy[self.horizon-t-1,s]=np.zeros(self.action_2_num)
                    self.strategy[self.horizon-t-1,s,action]=1
                else:
                    self.V[self.horizon-t-1,s] = max(np.dot(self.R[s, a1] + np.dot(self.T[s, a1], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a1 in range(self.action_1_num))
                    action=np.argmax(np.dot(self.R[s, a1] + np.dot(self.T[s, a1], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a1 in range(self.action_1_num))
                    self.strategy[self.horizon-t-1,s]=np.zeros(self.action_1_num)
                    self.strategy[self.horizon-t-1,s,action]=1
                '''
                if np.sum(self.strategy[self.horizon-t-1,s])==0:
                    print("strategy is zero",self.horizon-t-1,s)
                    print(self.strategy[self.horizon-t-1,s,0])
                    #print(action,'r')
                '''
    
    def choose_action(self, s, t):
        '''
        if np.sum(self.strategy[t,s])==0:
            print("Warning: strategy is zero",t,s)
        '''
        self.strategy[t,s] /= np.sum(self.strategy[t,s])
        if not self.is_minimizer:
            action = np.random.choice(range(self.action_1_num), p=self.strategy[t,s])
        else:
            action = np.random.choice(range(self.action_2_num), p=self.strategy[t,s])
        return action


    def value(self,s):
      return self.V[0,s]

    def get_strategy(self):
      return self.strategy

    def compute_measures(self, T):
        """
        Compute various distance measures between two tensors self_T and T.

        Parameters:
            self_T (np.ndarray): The first tensor, shape (81, 4, 4, 81).
            T (np.ndarray): The second tensor, shape (81, 4, 4, 81).

        Returns:
            dict: A dictionary containing the computed measures.
        """
        self_T = self.T
        # Ensure tensors have the correct dimensions
        assert self_T.shape == T.shape == (self.state_num, self.action_1_num, self.action_2_num, self.state_num), "Tensors must have shape (81, 4, 4, 81)."

        # Compute element-wise difference
        diff = self_T - T

        # Reshape the difference tensor into a matrix of shape (81*4*4, 81)
        diff_matrix = diff.reshape(self.state_num * self.action_1_num * self.action_2_num, self.state_num)

        # 1. Frobenius Norm
        frobenius_norm = np.sqrt(np.sum(diff_matrix**2))

        # 2. L1 Norm (maximum column-wise deviation per reshaped matrix)
        l1_norm = np.max(np.sum(np.abs(diff_matrix), axis=0))

        # 3. L∞ Norm (maximum row-wise deviation per reshaped matrix)
        linf_norm = np.max(np.sum(np.abs(diff_matrix), axis=1))

        return frobenius_norm, l1_norm, linf_norm


class Agent_fp_true(Agent):
    def __init__(self, game, horizon, is_minimizer=False,larger_prior=False,seed=292):
        super().__init__(game, horizon, is_minimizer,seed)
        if is_minimizer:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_1_num))/self.action_1_num for i in range(horizon) ]
            #self.V=utility.solve_value_by_strategy(self.T,game, horizon,maximal=False)
            self.opponent_history=[ np.ones((self.state_num,self.action_1_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_2_num))
        else:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_2_num))/self.action_2_num for i in range(horizon) ]
            #self.V=utility.solve_value_by_strategy(game, horizon,maximal=True)
            self.opponent_history=[ np.ones((self.state_num,self.action_2_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_1_num))
        self.R=game.R
        self.T = game.T
        self.V=np.zeros((self.horizon+1,self.state_num))
   
    def update_prior(self,next_s,s,a_1,a_2,tau):
        # Update opponent's history and strategy
        if self.is_minimizer:
            self.opponent_history[tau][s,a_1]+=1
            for state in range(self.state_num):
                for a in range(self.action_1_num):
                    self.opponent_strategy[tau][state,a] = (self.opponent_history[tau][state,a]) / np.sum(self.opponent_history[tau][state])
        else:
            self.opponent_history[tau][s,a_2]+=1
            for state in range(self.state_num):
                for a in range(self.action_2_num):
                    self.opponent_strategy[tau][state,a] = (self.opponent_history[tau][state,a]) / np.sum(self.opponent_history[tau][state])

    def solve_value(self):
        for t in range(self.horizon):  # Run value iteration
            for s in range(self.state_num):
                if self.is_minimizer:
                    self.V[self.horizon-t-1,s] = min(np.dot(self.R[s, :,a2] + np.dot(self.T[s,:, a2], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a2 in range(self.action_2_num))
                    action=np.argmin(np.dot(self.R[s, :,a2] + np.dot(self.T[s,:, a2], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a2 in range(self.action_2_num))
                    self.strategy[self.horizon-t-1,s]=np.zeros(self.action_2_num)
                    self.strategy[self.horizon-t-1,s,action]=1
                else:
                    self.V[self.horizon-t-1,s] = max(np.dot(self.R[s, a1] + np.dot(self.T[s, a1], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a1 in range(self.action_1_num))
                    action=np.argmax(np.dot(self.R[s, a1] + np.dot(self.T[s, a1], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a1 in range(self.action_1_num))
                    self.strategy[self.horizon-t-1,s]=np.zeros(self.action_1_num)
                    self.strategy[self.horizon-t-1,s,action]=1
                '''
                if np.sum(self.strategy[self.horizon-t-1,s])==0:
                    print("strategy is zero",self.horizon-t-1,s)
                    print(self.strategy[self.horizon-t-1,s,0])
                    #print(action,'r')
                '''
    
    def choose_action(self, s, t):
        '''
        if np.sum(self.strategy[t,s])==0:
            print("Warning: strategy is zero",t,s)
        '''
        self.strategy[t,s] /= np.sum(self.strategy[t,s])
        if not self.is_minimizer:
            action = np.random.choice(range(self.action_1_num), p=self.strategy[t,s])
        else:
            action = np.random.choice(range(self.action_2_num), p=self.strategy[t,s])
        return action


    def value(self,s):
      return self.V[0,s]

    def get_strategy(self):
      return self.strategy

    


class Agent_sample(Agent):
    def __init__(self, game, horizon, is_minimizer=False,larger_prior=False,seed=292):
        super().__init__(game, horizon, is_minimizer,seed)
        if is_minimizer:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_1_num))/self.action_1_num for i in range(horizon) ]
            self.V=utility.solve_value_by_strategy(game, horizon,maximal=False)
            self.opponent_history=[ np.ones((self.state_num,self.action_1_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_2_num))
        else:
            self.opponent_strategy=[ np.ones((self.state_num,self.action_2_num))/self.action_2_num for i in range(horizon) ]
            self.V=utility.solve_value_by_strategy(game, horizon,maximal=True)
            self.opponent_history=[ np.ones((self.state_num,self.action_2_num)) for i in range(horizon) ]
            self.strategy=np.zeros((horizon,self.state_num,self.action_1_num))
        self.R=game.R
        self.T = np.zeros((self.state_num, game.action_1_num, game.action_2_num, self.state_num))  # Transition probabilities (initialized to zero)
        
    def update_T(self,T_1,T_2):
        for s in range(self.state_num):
          for a_1 in range(self.action_1_num):
              for a_2 in range(self.action_2_num):
                for s_next in range(self.state_num):
                  self.T[s,a_1,a_2,s_next] = T_1[s%9,a_1,s_next%9]*T_2[s//9,a_2,s_next//9]


    def sample_model(self):
        if self.larger_prior:
          for s in range(self.state_num):
            for a1 in range(self.action_1_num):
                for a2 in range(self.action_2_num):
                    for s_next in range(self.state_num):
                        self.T[s, a1,a2,s_next] = (self.prior[s, a1,a2,s_next]-1)/(np.sum(self.prior[s, a1,a2])-self.state_num)

        else:
            T_1=np.zeros((self.state_num_1,self.action_1_num,self.state_num_1))
            T_2=np.zeros((self.state_num_2,self.action_2_num,self.state_num_2))
            # Sample transition probabilities from the Dirichlet prior for each state-action pair
            for s in range(self.state_num_1):
                for a1 in range(self.action_1_num):
                    T_1[s, a1] = self.rng.dirichlet(self.prior_1[s, a1])
            for s in range(self.state_num_2):
                for a2 in range(self.action_2_num):
                    T_2[s, a2] = self.rng.dirichlet(self.prior_2[s, a2])
            self.update_T(T_1,T_2)


    def update_prior(self,next_s,s,a_1,a_2,tau):
        if self.larger_prior:
            self.prior[s,a_1,a_2,next_s]+=1
        else:
            self.prior_1[s%9,a_1,next_s%9]+=1
            self.prior_2[s//9,a_2,next_s//9]+=1
        # Update opponent's history and strategy
        if self.is_minimizer:
            self.opponent_history[tau][s,a_1]+=1
            for state in range(self.state_num):
                for a in range(self.action_1_num):
                    self.opponent_strategy[tau][state,a] = (self.opponent_history[tau][state,a]) / np.sum(self.opponent_history[tau][state])
        else:
            self.opponent_history[tau][s,a_2]+=1
            for state in range(self.state_num):
                for a in range(self.action_2_num):
                    self.opponent_strategy[tau][state,a] = (self.opponent_history[tau][state,a]) / np.sum(self.opponent_history[tau][state])

    def solve_value(self):
        for t in range(self.horizon):  # Run value iteration
            for s in range(self.state_num):
                if self.is_minimizer:
                    self.V[self.horizon-t-1,s] = min(np.dot(self.R[s, :,a2] + np.dot(self.T[s,:, a2], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a2 in range(self.action_2_num))
                    action=np.argmin(np.dot(self.R[s, :,a2] + np.dot(self.T[s,:, a2], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a2 in range(self.action_2_num))
                    self.strategy[self.horizon-t-1,s]=np.zeros(self.action_2_num)
                    self.strategy[self.horizon-t-1,s,action]=1
                else:
                    self.V[self.horizon-t-1,s] = max(np.dot(self.R[s, a1] + np.dot(self.T[s, a1], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a1 in range(self.action_1_num))
                    action=np.argmax(np.dot(self.R[s, a1] + np.dot(self.T[s, a1], self.V[self.horizon-t]),self.opponent_strategy[t][s]) for a1 in range(self.action_1_num))
                    self.strategy[self.horizon-t-1,s]=np.zeros(self.action_1_num)
                    self.strategy[self.horizon-t-1,s,action]=1
                '''
                if np.sum(self.strategy[self.horizon-t-1,s])==0:
                    print("strategy is zero",self.horizon-t-1,s)
                    print(self.strategy[self.horizon-t-1,s,0])
                    #print(action,'r')
                '''
    
    def choose_action(self, s, t):
        '''
        if np.sum(self.strategy[t,s])==0:
            print("Warning: strategy is zero",t,s)
        '''
        self.strategy[t,s] /= np.sum(self.strategy[t,s])
        if not self.is_minimizer:
            action = np.random.choice(range(self.action_1_num), p=self.strategy[t,s])
        else:
            action = np.random.choice(range(self.action_2_num), p=self.strategy[t,s])
        return action


    def value(self,s):
      return self.V[0,s]

    def get_strategy(self):
      return self.strategy

    def compute_measures(self, T):
        """
        Compute various distance measures between two tensors self_T and T.

        Parameters:
            self_T (np.ndarray): The first tensor, shape (81, 4, 4, 81).
            T (np.ndarray): The second tensor, shape (81, 4, 4, 81).

        Returns:
            dict: A dictionary containing the computed measures.
        """
        self_T = self.T
        # Ensure tensors have the correct dimensions
        assert self_T.shape == T.shape == (self.state_num, self.action_1_num, self.action_2_num, self.state_num), "Tensors must have shape (81, 4, 4, 81)."

        # Compute element-wise difference
        diff = self_T - T

        # Reshape the difference tensor into a matrix of shape (81*4*4, 81)
        diff_matrix = diff.reshape(self.state_num * self.action_1_num * self.action_2_num, self.state_num)

        # 1. Frobenius Norm
        frobenius_norm = np.sqrt(np.sum(diff_matrix**2))

        # 2. L1 Norm (maximum column-wise deviation per reshaped matrix)
        l1_norm = np.max(np.sum(np.abs(diff_matrix), axis=0))

        # 3. L∞ Norm (maximum row-wise deviation per reshaped matrix)
        linf_norm = np.max(np.sum(np.abs(diff_matrix), axis=1))

        return frobenius_norm, l1_norm, linf_norm


