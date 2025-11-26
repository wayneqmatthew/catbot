import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

## utils
def extract_positions_bot(state):
    bot_pos = state / 100

    return int(bot_pos)
    
def extract_positions_cat(state):
    cat_pos = state % 100

    return int(cat_pos)

def manhattan_distance(bot_pos, cat_pos):
    start_x = int(bot_pos / 10)
    start_y = int(bot_pos % 10)
    goal_x = int(cat_pos / 10)
    goal_y = int(cat_pos % 10)

    distance = abs(start_x - goal_x) + abs(start_y - goal_y)
    
    return distance 

def cat_near_edge(new_cat_pos):
    cat_x = int(new_cat_pos / 10)
    cat_y = int(new_cat_pos % 10)

    for i in range (4):
        if i == 0:
            x, y = 0,0

        elif i == 1:
            x, y = 0, 6

        elif i == 2:
            x, y = 6, 0

        elif i == 3:
            x, y = 6, 6

        for j in range (x, x+2):
            for k in range (y, y+2):
                if cat_x == j and cat_y == k:
                    return True
    
    return False



def calculate_reward(old_state, new_state, done, step_count):
    #Get positions
    old_bot_pos = extract_positions_bot(old_state)
    old_cat_pos = extract_positions_cat(old_state)
    new_bot_pos = extract_positions_bot(new_state)
    new_cat_pos = extract_positions_cat(new_state)

    #calculate distance
    old_distance = manhattan_distance(old_bot_pos, old_cat_pos)
    new_distance = manhattan_distance(new_bot_pos, new_cat_pos)

    reward = -0.5 #base reward per step

    #reward for catching the cat
    if done and new_distance == 0:
        return 100
    
    #reward for over steps
    if step_count >= 60:
        return -50

    #reward for distance
    if new_distance < old_distance:
        if old_distance > 4:
            reward += 2 #since some cats are harder to chase when we get closer we don't reward being super close
        elif old_distance > 1:
            reward += 1
        else:
            reward -= 1

    if cat_near_edge(new_cat_pos) and new_distance < 4:
        reward += 3

    return reward
    


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    learning_rate = 0.1
    discount_factor = 0.95

    ## always make sure that exploration_rate never goes to zero, atleast 5%
    exploration_rate = 0.3
    epsilon_decay = 0.995
    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
               
        






























        
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table