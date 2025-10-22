#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:34:49 2022

@author: ntnu
"""

import numpy as np 
import random 
import time 
import os
import math
import gym
from gym.envs.classic_control import rendering

class Basketball_environment(gym.Env):

    def __init__(self, W, H, nb_opp):
        
        # defining parameters
        self.H = H
        self.W=W
        self.nb_opp = nb_opp
        
        # Visualisation
        self.viewer = None

        # Setting the positions
        self.ball_pos = [0, self.H//2 -1] 
        self.basket_pos = [self.W-1, self.H//2 -1]
        
        # Creating a list with all the possible positions on the field
        pos = []
        for i in range(self.W):
          for j in range(self.H):
            pos.append([i,j])
            
        # We randomly initialize the position of the robot which can't be on 
        # the basket or the ball
        pos.remove(self.ball_pos)
        pos.remove(self.basket_pos)
        r = random.randint(0, self.H * self.W -3)
        self.robot_pos = pos[r]
        pos.remove(self.robot_pos)
        
        # We similarly initialize the positions of the opponents given their 
        # number
        self.opp_pos = []
        for i in range(self.nb_opp):
          r = random.randint(0, self.H * self.W -4 -i)
          self.opp_pos.append(pos[r])
          pos.remove(pos[r])

        # Initialisation of the set of possible actions:
        self.actions = {0: "UP", 1: "LEFT", 2: "DOWN", 3: "RIGHT", 4: "DRIBBLE-UP", 5: "DRIBBLE-LEFT", 6: "DRIBBLE-DOWN", 7: "DRIBBLE-RIGHT", 8: "SHOOT"}

    # Function that gives a binary ouput indicating whether or not the robot
    # is in possession of the ball>
    def holds_ball(self):
        if self.robot_pos[0] == self.ball_pos[0] and self.robot_pos[1] == self.ball_pos[1]:
            holds_the_ball = 1
        else:
            holds_the_ball = 0  
        return holds_the_ball
    
    # Function that given the action selected would target an occupied cell
    # Binary function
    def opponent_encounter(self, movement):
        for i in range(self.nb_opp):
            if self.opp_pos[i][0] == (self.robot_pos[0] + movement[0]) and self.opp_pos[i][1] == (self.robot_pos[1] + movement[1]):
                encounters_opponent = 1 
                return encounters_opponent
            else:
                encounters_opponent = 0    
        return encounters_opponent

    # Boolean function that indicates if the robot has left the field.
    def leaves_court(self):
        if self.robot_pos[0] < 0 or self.robot_pos[0] > (self.W - 1) or self.robot_pos[1] < 0 or self.robot_pos[1] > (self.H - 1):
            return True
        else:
            return False
    
    # Reset function to reinitiate the environment for each new episode. 
    def reset(self):
        self.ball_pos = [0, self.H//2 -1] 
        pos = []
        for i in range(self.W):
          for j in range(self.H):
            pos.append([i,j])
        pos.remove(self.ball_pos)
        pos.remove(self.basket_pos)
        for opp in self.opp_pos:
            pos.remove(opp)
        r = random.randint(0,self.H * self.W -3 - self.nb_opp)
        self.robot_pos = pos[r]

    # step function 
    def step(self, action):
        
        # initialisation of the ouputs
        done = False
        reward = 0
        holds_the_ball = self.holds_ball()
        scored=0
        left_field = 0
        took_a_shot =0
        
        #70% chance of moving one step in the chosen direction, otherwise 2.
        r = random.random()
        if r < 0.7: 
            step_size = 1
        else:
            step_size = 2
            
        # If it it an action of MOVEMENT
        if action < 4:
            
            if self.actions[action] == "UP":  
                robot_movement = [0, step_size]
            
            elif self.actions[action] == "DOWN":
                robot_movement = [0, -step_size]
                
            elif self.actions[action] == "RIGHT":
                robot_movement = [step_size, 0]
                
            elif self.actions[action] == "LEFT":
                robot_movement = [-step_size, 0]
            
            encounters_opponent = self.opponent_encounter(robot_movement)
            # The command fails if he has the ball and moves instead of 
            # dribbling or if he moves into an occupied target cell
            command_success = (1 - holds_the_ball) * (1 - encounters_opponent)
            illegal_move = 1 - command_success
            # reward = illegal_move*(-2) # additionnal reward to penalize 
            # illegal moves
            # change the position
            self.robot_pos[0] += command_success * robot_movement[0]
            self.robot_pos[1] += command_success * robot_movement[1]
         
        # If it it an action of DRIBBLING
        elif action < 8:
            
            if self.actions[action] == "DRIBBLE-UP":
                robot_movement = [0, step_size* holds_the_ball] 
    
            elif self.actions[action] == "DRIBBLE-DOWN":
                robot_movement = [0, -step_size* holds_the_ball]
                
            elif self.actions[action] == "DRIBBLE-RIGHT":
                robot_movement = [step_size* holds_the_ball, 0]
                
            elif self.actions[action] == "DRIBBLE-LEFT":
                robot_movement = [-step_size* holds_the_ball, 0]
            
            # Command fails if the robot doesn't have the ball prior
            # If he does have the ball but encounters an opponent while 
            # dribbling, not only does the command fail, but he get penalized
            # -5
            encounters_opponent = self.opponent_encounter(robot_movement)
            command_success = 1 - encounters_opponent
            illegal_move = 1 - command_success*(holds_the_ball)
            
            reward = -5 * (1 - command_success) * (holds_the_ball) # - 2 * (1 - holds_the_ball) # additionnal reward to penalize illegal moves
            
            # change the positions
            self.robot_pos[0] += command_success * robot_movement[0]
            self.robot_pos[1] += command_success * robot_movement[1]
            self.ball_pos[0] += command_success * robot_movement[0] - encounters_opponent * holds_the_ball * self.ball_pos[0]
            self.ball_pos[1] += command_success * robot_movement[1] + encounters_opponent * holds_the_ball * (self.H/2 - 1 - self.ball_pos[1])
    

        elif self.actions[action] == "SHOOT":

            # calculate the distance to the basket to determine the probability of scoring
            dist = np.sqrt((self.basket_pos[0]-self.robot_pos[0])**2 + (self.basket_pos[1]-self.robot_pos[1])**2)
            r = random.random() + 1 - holds_the_ball # r greater than 1 if the opponent doesn't have the ball prior.
            # if the robot scores, it ends the episode
            done = (dist <= 1 and r < 0.9) or (dist <= 3 and r < 0.66) or (dist <= 4 and r < 0.1) 
            reward = (dist <= 1)*(r < 0.9)*2 + (dist > 1)*(dist <= 3)*(r < 0.66)*10 + (dist > 3)*(dist <= 4)*(r < 0.1)*30
            self.ball_pos = [(1 - holds_the_ball)*self.ball_pos[0] + holds_the_ball * (dist <= 4)*(reward == 0)*((self.W * 4)//5 - 1) + (reward != 0)*self.basket_pos[0], (1 - holds_the_ball)*self.ball_pos[0] + holds_the_ball * (dist <= 4)*(reward == 0)*(self.H//2 - 1) + (reward != 0)*self.basket_pos[1]] 
            scored = done * done
            illegal_move = 1 - (dist <= 4)*holds_the_ball
            took_a_shot = 1 - illegal_move
            
            # To penalize an illegal shot attempt or reward the creation of a legal shooting opportunity that didn't score
            # reward = illegal_move*(-2) + (1-illegal_move)*(1 - scored)*(4/((dist < 1)*1+(dist >= 1)*4))
            
        # End of the episode if the robot leaves the pitch, and a penalty of -100
        if self.leaves_court():
            done = True
            reward = -100
            left_field = 1
        
        
        state = np.array([self.robot_pos[0], self.robot_pos[1], self.ball_pos[0], self.ball_pos[1]])
        grabbed_the_ball = self.holds_ball()
        
        '''
        # Additional reward for the agent if he gets the ball
        if grabbed_the_ball - holds_the_ball > 0:
            reward = 15
        '''
        
        # On top of the neural network inputs, done and the reward value, all the other ouputs are for further analysis.
        return state, done, reward, grabbed_the_ball, scored, left_field, illegal_move, took_a_shot

    # To visualize the episodes
    def render(self, mode = 'human'):

        # initialize the screen dimensions
        block_size = 500//max(self.W, self.H)
        viewer_size_x = block_size*self.W
        viewer_size_y = block_size*self.H
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(viewer_size_x, viewer_size_y)
            
            # We first draw the court as a grid
            for i in range(1,self.W):
                lineGeom = self.viewer.draw_line((i * block_size, 0), (i * block_size, viewer_size_y))
                self.viewer.add_geom(lineGeom)   
            for i in range(1,self.H):
                lineGeom = self.viewer.draw_line(start=(0, i * block_size), end=(viewer_size_x, i * block_size))
                self.viewer.add_geom(lineGeom)
            
            # We draw the basketball
            ball = self.viewer.draw_circle(block_size/2)
            self.balltrans = rendering.Transform()
            ball.add_attr(self.balltrans)
            ball.set_color(190/255,164/255,39/255)
            self.viewer.add_geom(ball)
            
            
            # We draw the basket
            basket = self.viewer.draw_circle(block_size/2)
            self.baskettrans = rendering.Transform((viewer_size_x - block_size/2, viewer_size_y/2 - block_size/2))
            basket.add_attr(self.baskettrans)
            basket.set_color(59/255, 210/255, 139/255)
            self.viewer.add_geom(basket)
    
            
            # We draw the robot
            robot = rendering.FilledPolygon([(-block_size/2, -block_size/2), (-block_size/2, block_size/2), (block_size/2, block_size/2), (block_size/2, -block_size/2)])
            self.robottrans = rendering.Transform((self.robot_pos[0]*block_size + block_size/2, self.robot_pos[1]*block_size + block_size/2))
            robot.add_attr(self.robottrans)
            robot.set_color(186/255, 47/255, 115/255)
            self.viewer.add_geom(robot)
    
    
            # make the Opponent
            # the size and shape is equal the agent
            self.opptrans = []
            for oppPosition in self.opp_pos:
                opp = rendering.FilledPolygon([(-block_size/2, -block_size/2), (-block_size/2, block_size/2), (block_size/2, block_size/2), (block_size/2, -block_size/2)])
                opptran = rendering.Transform((oppPosition[0]*block_size + block_size/2, oppPosition[1]*block_size + block_size/2))
                opp.add_attr(opptran)
                opp.set_color(22/255,14/255,31/255)
                self.opptrans.append(opptran)
                self.viewer.add_geom(opp)

        
        # update agent transform
        self.robottrans.set_translation(self.robot_pos[0]*block_size + block_size/2, self.robot_pos[1]*block_size + block_size/2)

        # update basketball transform
        self.balltrans.set_translation(self.ball_pos[0]*block_size + block_size/2, self.ball_pos[1]*block_size + block_size/2)
    
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    






    
