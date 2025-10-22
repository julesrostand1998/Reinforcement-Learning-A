#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import time
from itertools import count
from environment import Basketball_environment
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# generate environment
env = Basketball_environment(W=9, H=6, nb_opp=5)

# discount factor
gamma = 0.9

# Create the policy network
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 9)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

# Initialize a policy and policy loss optimization parameters.
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

# Randmly select an action given the current weights.
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

# Update the weights after the end of the episode
def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if len(returns) == 1:
        returns_std = 0
    else:
        returns_std = returns.std()
    returns = (returns - returns.mean()) / (returns_std + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    
    '''
    #  initialisation of lists for further analysis (see report)
    nb_steps=[]
    grabbing_ball=[]
    scoring=[]
    nb_of_shots=[]
    prop_illegal_moves=[]
    leaving_field=[]
    rew=[]
    '''
    
    for i_episode in range(101):
        print('ep', i_episode)
        state, ep_reward = np.array( [env.robot_pos[0], env.robot_pos[1], env.ball_pos[0], env.ball_pos[1]] ), 0
        '''
        
        grabbed_the_ball_prev = 0
        nb_of_illegal_moves = 0
        shots_taken = 0
        '''
        
        for t in range(1, 101):  
            # Initialize visualisation as long as they are episodes
            env.render()
            # Select action
            action = select_action(state)
            # update after performing the action. 
            state, done, reward, grabbed_the_ball, scored, left_field, illegal_move, took_a_shot = env.step(action)
            '''
            # Update of parameters for report analysis
            grabbed_the_ball_prev = max(grabbed_the_ball_prev, grabbed_the_ball)
            nb_of_illegal_moves += illegal_move
            shots_taken += took_a_shot
            '''
            policy.rewards.append(reward)
            ep_reward += reward
            time.sleep(0.01) 
            # If the robot leaves the field or scores a point, the episode ends.
            if done:
                break
            
        '''
        # Update the tabs so they are ready for plots
        nb_steps.append(t)
        grabbing_ball.append(grabbed_the_ball_prev)
        scoring.append(scored)
        nb_of_shots.append(shots_taken)
        leaving_field.append(left_field)
        rew.append(ep_reward)
        prop_illegal_moves.append(nb_of_illegal_moves/t)
        '''
        
        time.sleep(0.05)
        
        # Weights update
        finish_episode()
        
        # New episode
        env.reset()
    # We end the visualisation at the end of the training.
    env.close()
    
    '''
    # Plots for analysis
    episodes = [i for i in range(101)]
    
    nb_steps = [sum(nb_steps[(i-50):i])/len(nb_steps[(i-50):i]) for i in range(50,len(nb_steps))]
    grabbing_ball = [sum(grabbing_ball[(i-50):i])/len(grabbing_ball[(i-50):i]) for i in range(50,len(grabbing_ball))]
    scoring = [sum(scoring[(i-50):i])/len(scoring[(i-50):i]) for i in range(50,len(scoring))]
    nb_of_shots = [sum(nb_of_shots[(i-50):i])/len(nb_of_shots[(i-50):i]) for i in range(50,len(nb_of_shots))]
    leaving_field = [sum(leaving_field[(i-50):i])/len(leaving_field[(i-50):i]) for i in range(50,len(leaving_field))]
    rew = [sum(rew[(i-50):i])/len(rew[(i-50):i]) for i in range(50,len(rew))]
    prop_illegal_moves = [sum(prop_illegal_moves[(i-50):i])/len(prop_illegal_moves[(i-50):i]) for i in range(50,len(prop_illegal_moves))]
    
    fig = plt.figure(figsize = (30, 20))
    
    ax1 = fig.add_subplot(3,2,1)
    ax1.plot(episodes[50:], nb_steps, '-', color='red', label=f"gamma = {0.9}")
    plt.title("evolution of the total number of steps per episode")
    plt.xlabel('episode number')
    plt.ylabel('total number of steps')
    ax1.legend()
    
    ax2 = fig.add_subplot(3,2,2)
    ax2.plot(episodes[50:], grabbing_ball, '-', color='red', label=f"gamma = {0.9}")
    plt.title("evolution of the learning of getting to the ball first per episode")
    plt.xlabel('episode number')
    plt.ylabel('grabbed the ball')
    ax2.legend()
    
    ax3 = fig.add_subplot(3,2,3)
    ax3.plot(episodes[50:], scoring, 'o', color='blue', label = "scored")
    ax3.plot(episodes[50:], nb_of_shots, '-', color='red', label = "shots_taken")
    plt.title("evolution of the learning to score and create shooting opportunities per episode ")
    plt.xlabel('episode number')
    plt.ylabel('scored and shots')
    ax3.legend()
    
    ax4 = fig.add_subplot(3,2,4)
    ax4.plot(episodes[50:], leaving_field, '-', color='red', label=f"gamma = {0.9}")
    plt.title("evolution of the learning of avoiding to leave the field per episode")
    plt.xlabel('episode number')
    plt.ylabel('left the field')
    ax4.legend()
    
    ax5 = fig.add_subplot(3,2,5)
    ax5.plot(episodes[50:], rew, '-', color='red', label=f"gamma = {0.9}")
    plt.title("evolution of the total reward per episode")
    plt.xlabel('episode number')
    plt.ylabel('total reward')
    ax5.legend()
    
    ax6 = fig.add_subplot(3,2,6)
    ax6.plot(episodes[50:], prop_illegal_moves, '-', color='red', label=f"gamma = {0.9}")
    plt.title("evolution of the proportion of illegal actions per episode")
    plt.xlabel('episode number')
    plt.ylabel('proportion of illegal actions')
    ax6.legend()
    '''
    
if __name__ == '__main__':
    main()