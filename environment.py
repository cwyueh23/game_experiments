import numpy as np
import matplotlib.pyplot as plt
from agent import *
from game import *
from utility import *

def play_eq(agent_1,agent_2,game,tau=10,episodes=1000,filename='',test_best=0):

    state_num=game.state_num
    init_prob=np.ones(state_num)/state_num

    V_optimal_max = solve_value(game.T,game,tau,maximum=True)  # Optimal value for maximizing agent
    V_optimal_min = solve_value(game.T,game,tau,maximum=False)  # Optimal value for minimizing agent

    # Lists to store records and regret
    record = []
    regret_agent_1 = []
    regret_agent_2 = []

    frobenius_norm = []
    l1_norm = []
    linf_norm = []

    # Run episodes and track agent actions, rewards, and regret
    for episode in range(episodes):
        # Agents sample their model for the episode

        agent_1.sample_model()
        agent_2.sample_model()
        # Agents solve for the value function
        if test_best==0:
            agent_1.solve_value()
            agent_2.solve_value()
        elif test_best==1:
            agent_1.solve_value()
            agent_2.solve_value(agent_1.get_strategy())
        elif test_best==2:
            agent_2.solve_value()
            agent_1.solve_value(agent_2.get_strategy())
            
        #s = 72  # Starting state
        s=np.random.choice(range(state_num), p=init_prob)
        s_0=s
        record_ep = []  # Record for this episode
        total_reward_agent_1 = 0
        total_reward_agent_2 = 0

        # Timestep loop within an episode
        for t in range(tau):
            # Both agents choose actions
            a_1 = agent_1.choose_action(s, t)
            a_2 = agent_2.choose_action(s, t)

            # Calculate reward and transition to the next state
            reward = game.r(s, a_1, a_2)
            next_s = game.Transition(s, a_1, a_2)

            # Update agents' prior knowledge based on the transition
            agent_1.update_prior(next_s, s, a_1, a_2,t)
            agent_2.update_prior(next_s, s, a_1, a_2,t)

            # Record the state, actions, and rewards for this timestep
            record_ep.append((s, a_1, a_2, reward))

            # Accumulate rewards for regret analysis
            total_reward_agent_1 += reward
            total_reward_agent_2 += -reward  # Zero-sum game, opposite rewards

            # Move to the next state
            s = next_s

        # Store the episode record
        record.append(record_ep)

        # Store the episode record
        record.append(record_ep)
        measures = agent_1.compute_measures(game.T)
        frobenius_norm.append(measures[0])
        l1_norm.append(measures[1])
        linf_norm.append(measures[2])

        # Calculate and store the regret for this episode
        regret_agent_1.append(V_optimal_max[0, s_0] - total_reward_agent_1)
        regret_agent_2.append(-total_reward_agent_2 - V_optimal_min[0, s_0])

        #regret_exp_1.append(V_optimal_max[0, 72] - agent_1.value(72))
        #regret_exp_2.append(-agent_2.value(72) - V_optimal_min[0, 72])

    # Display the recorded episode data (optional)
    #print(record)

    # Plot the regret for both agents over episodes
    plt.figure()
    plt.plot(range(episodes), regret_agent_1, label='Agent 1 Regret')
    plt.plot(range(episodes), regret_agent_2, label='Agent 2 Regret')
    plt.xlabel('Episode')
    plt.ylabel('Regret')
    plt.title('Regret Over Episodes for Agent 1 and Agent 2')
    plt.legend()
    plt.savefig(filename+'_regret_eq.png')


    regret_cumulative_1=[]
    regret_cumulative_2=[]
    for i in range(episodes):
        regret_cumulative_1.append(sum(regret_agent_1[:i])/(i+1))
        regret_cumulative_2.append(sum(regret_agent_2[:i])/(i+1))

    plt.figure()
    plt.plot(range(episodes), regret_cumulative_1, label='Agent 1 Time-Averaged Regret')
    plt.plot(range(episodes), regret_cumulative_2, label='Agent 2 Time_Averaged Regret')

    plt.xlabel('Episode')
    plt.ylabel('Time-Averaged Regret')
    plt.title('Time-Averaged Regret Over Episodes for Agent 1 and Agent 2')
    plt.legend()
    plt.savefig(filename+'_regret_eq_cumulative.png')


    print(regret_cumulative_1[-1])
    print(regret_cumulative_2[-1])

    # Plot all measures over episodes
    plt.figure()
    plt.plot(range(episodes), frobenius_norm, label='Frobenius Norm')
    plt.plot(range(episodes), l1_norm, label='L1 Norm')
    plt.plot(range(episodes), linf_norm, label='L∞ Norm')

    plt.xlabel('Episode')
    plt.ylabel('Measure Value')
    plt.title('Distance Measures Over Episodes')
    plt.legend()
    plt.savefig(filename+'_distance_eq.png')
    plt.show()

    return regret_cumulative_1,regret_cumulative_2

def play_single(agent_1,agent_2,game,tau=10,episodes=1000,filename='',test_best=0):


    state_num=game.state_num
    init_prob=np.ones(state_num)/state_num

    V_optimal_max = solve_value(game.T,game,tau,maximum=True)  # Optimal value for maximizing agent
    V_optimal_min = solve_value(game.T,game,tau,maximum=False)  # Optimal value for minimizing agent

    # Lists to store records and regret
    record = []
    regret_agent_1 = []
    regret_agent_2 = []

    frobenius_norm = []
    l1_norm = []
    linf_norm = []
    # Run episodes and track agent actions, rewards, and regret
    for episode in range(episodes):
        # Agents sample their model for the episode
        agent_1.sample_model()
        agent_2.sample_model()

        # Agents solve for the value function and determine strategies
        if test_best==0:
            agent_1.solve_value()
            agent_2.solve_value()
        elif test_best==1:
            agent_1.solve_value()
            agent_2.solve_value(agent_1.get_strategy())
        elif test_best==2:
            agent_2.solve_value()
            agent_1.solve_value(agent_2.get_strategy())

        #s = 72  # Starting state
        s=np.random.choice(range(state_num), p=init_prob)
        s_0=s
        record_ep = []  # Record for this episode
        total_reward_agent_1 = 0
        total_reward_agent_2 = 0

        # Timestep loop within an episode
        for t in range(tau):
            # Both agents choose actions
            a_1 = agent_1.choose_action(s, t)
            a_2 = agent_2.choose_action(s, t)

            # Calculate reward and transition to the next state
            reward = game.R[s, a_1, a_2]
            next_s = game.Transition(s, a_1, a_2)

            # Update agents' prior knowledge based on the transition
            agent_1.update_prior(next_s, s, a_1, a_2,t)
            agent_2.update_prior(next_s, s, a_1, a_2,t)

            # Record the state, actions, and rewards for this timestep
            record_ep.append((s, a_1, a_2, reward))

            # Accumulate rewards for regret analysis
            total_reward_agent_1 += reward
            total_reward_agent_2 += -reward  # Zero-sum game, opposite rewards

            # Move to the next state
            s = next_s

        # Store the episode record
        record.append(record_ep)
        pi1=agent_1.get_strategy()
        pi2=agent_2.get_strategy()
        # Calculate and store the regret for this episode
        regret_agent_1.append(solve_value_by_strategy(game.T,game,tau,pi2)[0,s_0] - total_reward_agent_1)
        regret_agent_2.append(-total_reward_agent_2 - solve_value_by_strategy(game.T,game,tau,pi1,maximal=False)[0,s_0] )

        # Store the episode record
        record.append(record_ep)
        measures = agent_1.compute_measures(game.T)
        frobenius_norm.append(measures[0])
        l1_norm.append(measures[1])
        linf_norm.append(measures[2])

        #regret_exp_1.append(V_optimal_max[0, 72] - agent_1.value(72))
        #regret_exp_2.append(-agent_2.value(72) - V_optimal_min[0, 72])

        # Display the recorded episode data (optional)
        #print(record)
        
    fig1, ax1 = plt.subplots()
    # Plot the regret for both agents over episodes
    ax1.plot(range(episodes), regret_agent_1, label='Agent 1 Regret')
    ax1.plot(range(episodes), regret_agent_2, label='Agent 2 Regret')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Regret')
    ax1.set_title('Regret Over Episodes for Agent 1 and Agent 2')
    ax1.legend()
    fig1.savefig(filename+'_regret.png')

    regret_cumulative_1=[]
    regret_cumulative_2=[]
    for i in range(episodes):
        regret_cumulative_1.append(sum(regret_agent_1[:i])/(i+1))
        regret_cumulative_2.append(sum(regret_agent_2[:i])/(i+1))

    fig2, ax2 = plt.subplots()
    ax2.plot(range(episodes), regret_cumulative_1, label='Agent 1 Time-Averaged Regret')
    ax2.plot(range(episodes), regret_cumulative_2, label='Agent 2 Time-Averaged Regret')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Regret')
    ax2.set_title('Time-Averaged Regret Over Episodes for Agent 1')
    ax2.legend()
    fig2.savefig(filename+'_avgregret.png')
    #fig2.show()

    print(regret_cumulative_1[-1])
    print(regret_cumulative_2[-1])

    # Plot all measures over episodes
    plt.figure()
    plt.plot(range(episodes), frobenius_norm, label='Frobenius Norm')
    plt.plot(range(episodes), l1_norm, label='L1 Norm')
    plt.plot(range(episodes), linf_norm, label='L∞ Norm')

    plt.xlabel('Episode')
    plt.ylabel('Measure Value')
    plt.title('Distance Measures Over Episodes')
    plt.legend()
    plt.savefig(filename+'_distance.png')
    
    plt.show()
    
    return regret_cumulative_1,regret_cumulative_2
    regret_cumulative_2=[]