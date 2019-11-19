import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import os
MAX_PROFIT = 10000

import pickle

data = pd.read_excel(os.getcwd()+"/data/final_data.xlsx")
MAX_DEMAND = data['Ontario Demand'].max()
MAX_HOEP = data['HOEP'].max()
MAX_GENERATED_SUPPLY = data['Total Output'].max()
# previous_tariff = 0
# new_profit = 0

class SmartgridEnv(gym.Env):
    """A smart grid environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(SmartgridEnv, self).__init__()

        self.df = df
        self.reward_range = (0,MAX_PROFIT)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        48, 'Ontario Demand'].values / MAX_DEMAND,
            self.df.loc[self.current_step: self.current_step +
                        48, 'HOEP'].values / MAX_HOEP,
            self.df.loc[self.current_step: self.current_step +
                        48, 'TotalOutput'].values / MAX_GENERATED_SUPPLY,
        ])

        # Append additional data and scale each value to between 0-1
        # obs = np.append(frame, [[
        #     self.tariff / MAX_Tariff,
        #     self.demand / MAX_Demand,
        #     self.predicted_demand / MAX_Demand,
        #     self.company_HOEP / MAX_Company_HOEP,
        # ]], axis=0)

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:# previous profit is negative
            t = prev_t + amount * prev_t
            new_profit = (t*a - p*c - I|p-a|)

        elif action_type < 2: # previous profit is positive
            t = prev_t - amount * prev_t
            new_profit = (t*a - p*c - I|p-a|)

        

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = (1/(|new_profit|)) * delay_modifier
        ## done = new_profit <=  0
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state


        self.prev_t = 0
        self.new_profit = 0
        # Decide what is the initial state

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Ontario Demand'].values) - 48)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen

        prev_t = 0
        new_profit = 0
        # profit = 

        # print(f'Step: {self.current_step}')
        # print(f'Balance: {self.balance}')
        # print(
        #     f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        # print(
        #     f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        # print(
        #     f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        # print(f'Profit: {profit}')
