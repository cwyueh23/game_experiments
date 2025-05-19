import numpy as np
import random

class game:
    def __init___(self,seed=782):
        self.rng = np.random.default_rng(seed)
        self.state_num=1
        self.action_1_num=1
        self.action_2_num=1
        
        #Transition matrix
        self.T = np.ones([self.state_num,self.action_1_num,self.action_2_num,self.state_num])/self.state_num
        #Reward matrix
        self.R = np.zeros([self.state_num,self.action_1_num,self.action_2_num])
    
    def Transition(self,s,a_1,a_2):
        next_state=self.rng.choice(self.state_num, p=self.T[s,a_1,a_2])
        return next_state
    
    def r(self,s,a_1,a_2):
        return self.R[s,a_1,a_2]
    
# This code defines a grid game environment for two agents, where they can move in a 3x3 grid and receive rewards based on their actions and positions.
class grid_game(game):
    def __init__(self,seed=782):
        self.rng = np.random.default_rng(seed)
        self.state_num=81
        self.state_num_1=9
        self.state_num_2=9
        self.action_1_num=4
        self.action_2_num=4
        self.T = self.transition_matrix()
        self.R = self.r_matrix()

    def move(self,s,a):
        column=s//3
        row=s%3
        if a==0: #up
            row=(row+1)%3
        elif a==1: #right
            column=(column+1)%3
        elif a==2: #down
            row=(row-1)%3
        elif a==3: #left
            column=(column-1)%3
        return row+3*column

    def Transition(self,s,a_1,a_2):
        s_1=s%9
        s_2=s//9
        direction=self.rng.choice(4, 2, p=[0.7, 0.1, 0.1, 0.1])
        a_1=(a_1+direction[0])%4
        a_2=(a_2+direction[1])%4
        s_1=self.move(s_1,a_1)
        s_2=self.move(s_2,a_2)
        next_state=s_1+9*s_2
        return next_state
    
    def r(self,s,a_1,a_2):
        s_1=s%9
        s_2=s//9
        s1_next=self.move(s_1,a_1)
        s2_next=self.move(s_2,a_2)
        if s1_next==s2_next:
            if s1_next==4:
                reward=-4
            else:
                reward=-2
        else:
            column_1=s1_next//3
            row_1=s1_next%3
            column_2=s2_next//3
            row_2=s2_next%3
            reward=np.sqrt((column_1-column_2)**2+(row_1-row_2)**2)
            if s1_next==4:
                reward+=2
        return reward

    def r_matrix(self):
        reward=np.zeros((self.state_num,self.action_1_num,self.action_2_num))
        for s in range(self.state_num):
            for a_1 in range(self.action_1_num):
                for a_2 in range(self.action_2_num):
                    reward[s,a_1,a_2]=self.r(s,a_1,a_2)
        return reward

    # Create a transition matrix T
    def transition_matrix(self):
        # Initialize random transition probabilities for each state and action pair
        T = np.zeros((self.state_num, self.action_1_num, self.action_2_num, self.state_num))
        trans_prob=[0.7,0.1,0.1,0.1]
        for s in range(self.state_num):
            for a_1 in range(self.action_1_num):
                for a_2 in range(self.action_2_num):
                    # Generate random probabilities for transitions to all other states
                    for direction_1 in range(self.action_1_num):
                        for direction_2 in range(self.action_2_num):
                            adj_a1=(a_1+direction_1)%4
                            adj_a2=(a_2+direction_2)%4
                            s_1=s%9
                            s_2=s//9
                            s1_next=self.move(s_1,adj_a1)
                            s2_next=self.move(s_2,adj_a2)
                            s_next=9*s2_next+s1_next
                            T[s, a_1, a_2,s_next] = trans_prob[direction_1]*trans_prob[direction_2]

        return T
