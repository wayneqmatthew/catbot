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

    reward = -1.0 #base reward per step

    #reward for catching the cat
    if done and new_distance == 0:
        return 150
    
    #reward for over steps
    if step_count >= 60:
        return -50

    #reward for distance
    # if new_distance < old_distance:
    #     if old_distance > 4:
    #         reward += 2 #since some cats are harder to chase when we get closer we don't reward being super close
    #     elif old_distance > 1:
    #         reward += 1
    #     else:
    #         reward -= 1

    if new_distance < old_distance:
        if new_distance == 0:
            reward += 20
        elif new_distance == 1:
            reward += 10
        elif new_distance <= 3:
            reward += 8
        else:
            reward += 4

    elif new_distance > old_distance:
        reward -= 5

    if old_bot_pos == new_bot_pos:
        reward -= 5

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
    
    learning_rate = 0.2
    discount_factor = 0.95
    exploration_rate = 1.0 # always make sure that exploration_rate never goes to zero, atleast 5% || exploration rate starts at 1 for full exploration then decays over time
    epsilon_decay = 0.996
    minimum_epsilon = 0.05

    
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
               
        state, _ = env.reset()
        done = False
        step_count = 0

        ## new stuff for oscilliation. specificalkly initializing the previous states
        action_history = []
        state_history = [state]
        previous_state = state

        state_visit_count = {}
        state_visit_count[state] = 1

        #loop per episode
        while not done and step_count < 60:
            #if random value is less than exploration rate then we will do a random action "Exploration"
            if random.random() < exploration_rate: #.random() returns a random float between 0 and 1
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state]) #else we will base it of the values in the qtable "Exploitation"

            #take the action
            new_state, env_reward, done, truncated, info = env.step(action) #we never really use env_reward, truncated, and info its just part of the tuple that the step function returns
            step_count += 1

            reward = calculate_reward(state, new_state, done, step_count) #the env_reward we get fromm the action is zero which is why we solve for reward manually

            ## START
            if new_state in state_visit_count:
                state_visit_count[new_state] += 1
                reward -= (10 * state_visit_count[new_state])
            else:
                state_visit_count[new_state] = 1

            action_history.append(action)

            if len(action_history) >= 4:
                last_4 = action_history[-4:]
                if last_4[0] == last_4[2] and last_4[1] == last_4[3] and last_4[0] != last_4[1]:
                    reward -= 50

            if len(action_history) >= 6:
                last_6 = action_history[-6:]
                if(last_6[0] == last_6[2] == last_6[4] and last_6[1] == last_6[3] == last_6[5] and last_6[0] != last_6[1]):
                    reward -= 80

            if previous_state == new_state:
                reward -= 10

            ## punishes if it kept oscillating
            if len(state_history) >= 3:
                if new_state in state_history[-3:]:
                    reward -= 25

            if len(state_history) >= 4:
                if state_history[-2] == new_state and state_history[-4] == new_state:
                    reward -= 60

            state_history.append(new_state)
            previous_state = new_state

            ##END OF THIS FUCKINJG COPDE  HDAOIDHOIAHDO


            #needed variables for Q-Learning formula
            old_q_value = q_table[state][action] #old q value
            max_future_q = np.max(q_table[new_state])  #returns the highest value on the q_table

            new_q_value = old_q_value + learning_rate * (reward + discount_factor * max_future_q - old_q_value) #based on Q-Learning Formula 

            q_table[state][action] = new_q_value #we update the value on the qtable with the new recently computed value

            state = new_state #update states


        #apply the decay on the exploration rate
        exploration_rate = max(minimum_epsilon, exploration_rate * epsilon_decay)


    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################

    # If rendering is enabled, play an episode every 'render' episodes
    if render != -1 and (ep == 1 or ep % render == 0):
        viz_env = make_env(cat_type=cat_name)
        play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
        print('episode', ep)

    return q_table