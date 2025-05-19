import numpy as np
import matplotlib.pyplot as plt
from agent import *
from game import *
from utility import *

def play_eq(agent_1,agent_2,game,tau=10,episodes=1000):

    state_num=game.state_num
    init_prob=np.ones(state_num)/state_num

    V_optimal_max = solve_value_max(game,tau)  # Optimal value for maximizing agent
    V_optimal_min = solve_value_min(game,tau)  # Optimal value for minimizing agent

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
        agent_1.solve_value()
        agent_2.solve_value()

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
            agent_1.update_prior(next_s, s, a_1, a_2)
            agent_2.update_prior(next_s, s, a_1, a_2)

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
    plt.plot(range(episodes), regret_agent_1, label='Agent 1 Regret')
    plt.plot(range(episodes), regret_agent_2, label='Agent 2 Regret')
    plt.xlabel('Episode')
    plt.ylabel('Regret')
    plt.title('Regret Over Episodes for Agent 1 and Agent 2')
    plt.legend()
    plt.show()

    regret_cumulative_1=[]
    regret_cumulative_2=[]
    for i in range(episodes):
        regret_cumulative_1.append(sum(regret_agent_1[:i])/(i+1))
        regret_cumulative_2.append(sum(regret_agent_2[:i])/(i+1))

    plt.plot(range(episodes), regret_cumulative_1, label='Agent 1 Time-Averaged Regret')
    plt.plot(range(episodes), regret_cumulative_2, label='Agent 2 Time_Averaged Regret')

    plt.xlabel('Episode')
    plt.ylabel('Time-Averaged Regret')
    plt.title('Time-Averaged Regret Over Episodes for Agent 1 and Agent 2')
    plt.legend()
    plt.show()

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
    plt.show()



def play_single(agent_1,agent_2,game,tau=10,episodes=1000):

    state_num=game.state_num
    init_prob=np.ones(state_num)/state_num

    V_optimal_max = solve_value_max(game,tau)  # Optimal value for maximizing agent
    V_optimal_min = solve_value_min(game,tau)  # Optimal value for minimizing agent

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
        agent_1.solve_value()
        agent_2.solve_value()

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
            agent_1.update_prior(next_s, s, a_1, a_2)
            agent_2.update_prior(next_s, s, a_1, a_2)

            # Record the state, actions, and rewards for this timestep
            record_ep.append((s, a_1, a_2, reward))

            # Accumulate rewards for regret analysis
            total_reward_agent_1 += reward
            total_reward_agent_2 += -reward  # Zero-sum game, opposite rewards

            # Move to the next state
            s = next_s

        # Store the episode record
        record.append(record_ep)
        pi2=agent_2.get_strategy()
        # Calculate and store the regret for this episode
        regret_agent_1.append(solve_value_by_strategy(s_0,game,tau,pi2) - total_reward_agent_1)

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

    # Plot the regret for both agents over episodes
    plt.plot(range(episodes), regret_agent_1, label='Agent 1 Regret')
    #plt.plot(range(episodes), regret_agent_2, label='Agent 2 Regret')
    plt.xlabel('Episode')
    plt.ylabel('Regret')
    plt.title('Regret Over Episodes for Agent 1 and Agent 2')
    plt.legend()
    plt.show()

    regret_cumulative_1=[]
    for i in range(episodes):
        regret_cumulative_1.append(sum(regret_agent_1[:i])/(i+1))
    #regret_cumulative_2.append(sum(regret_agent_2[:i])/(i+1))

    plt.plot(range(episodes), regret_cumulative_1, label='Agent 1 Time-Averaged Regret')
    #plt.plot(range(episodes), regret_cumulative_2, label='Agent 2 cumulative Regret')

    plt.xlabel('Episode')
    plt.ylabel('Regret')
    plt.title('Time-Averaged Regret Over Episodes for Agent 1')
    plt.legend()
    plt.show()

    print(regret_cumulative_1[-1])
    #print(regret_cumulative_2[-1])

    # Plot all measures over episodes
    plt.figure()
    plt.plot(range(episodes), frobenius_norm, label='Frobenius Norm')
    plt.plot(range(episodes), l1_norm, label='L1 Norm')
    plt.plot(range(episodes), linf_norm, label='L∞ Norm')

    plt.xlabel('Episode')
    plt.ylabel('Measure Value')
    plt.title('Distance Measures Over Episodes')
    plt.legend()
    plt.show()