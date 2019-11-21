import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import os

MAX_STEPS = 20000

import pickle

data = pd.read_excel(os.getcwd()+"/data/RL_data.xlsx")
MAX_DEMAND = data['Actual Demand'].max()
MAX_PREDICTED_DEMAND = data['Predicted Demand'].max()
MAX_HOEP = data['Tariff'].max()
MAX_COMPANY_TARIFF = data['Company Tariff'].max()
NF = pow(10,5)
PROFIT = 1

class SmartGridEnv(gym.Env):
    """A smart grid environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(SmartGridEnv, self).__init__()

        self.df = df
        
        self.reward_range = (0,10/(abs(PROFIT)+1))

        # Actions of the format Increase x%, Decrease x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last fourty-eight prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4,48), dtype=np.float16)

    def _next_observation(self):
       
        obs = np.array([
            self.df.loc[self.current_step:self.current_step+47,'Actual Demand'] / MAX_DEMAND,
            self.df.loc[self.current_step:self.current_step+47,'Tariff'] / MAX_HOEP,
            self.df.loc[self.current_step:self.current_step+47,'Company Tariff'] / MAX_COMPANY_TARIFF, 
            np.repeat(self.df.loc[self.current_step,'Predicted Demand'] / MAX_PREDICTED_DEMAND, 48)               
        ])

        

        return obs

    def _take_action(self, action):
        action_type = action[0]
        amount = action[1]
        self.prev_t = self.df.iloc[self.current_step]['Tariff']
        actual = self.df.iloc[self.current_step]['Actual Demand']
        predicted = self.df.iloc[self.current_step]['Predicted Demand']
        company =  self.df.iloc[self.current_step]['Company Tariff'] 

        if action_type < 1:# previous profit is negative
            self.old_profit = (self.prev_t*actual - predicted*company - self.imbalance*abs(predicted-actual))
            self.prev_t = self.prev_t + amount * self.prev_t
            
        elif action_type < 2: # previous profit is positive
            self.old_profit = (self.prev_t*actual - predicted*company - self.imbalance*abs(predicted-actual))
            self.prev_t = self.prev_t - amount * self.prev_t
 

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Actual Demand'].values) - 48:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = (1/(abs(self.old_profit))+1) * delay_modifier
        
        done = abs(self.old_profit)<=0.004

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state


        self.prev_t = 0
        self.imbalance = 0.08
        self.initial_tariff = self.df.iloc[0]['Tariff']
        self.initial_actual_demand = self.df.iloc[0]['Actual Demand']
        self.initial_predicted_demand = self.df.iloc[0]['Predicted Demand']
        self.initial_company_tariff = self.df.iloc[0]['Company Tariff']

        self.old_profit = self.initial_tariff*self.initial_actual_demand- self.initial_predicted_demand*self.initial_company_tariff - self.imbalance*(abs(self.initial_predicted_demand-self.initial_actual_demand))
        # Decide what is the initial state

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Actual Demand'].values) - 48)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen


        print(f'Step: {self.current_step}')
        print(f'Tariff: {self.prev_t}')
        print(f'Profit: {self.old_profit}')
    ## done = new_profit <=  0