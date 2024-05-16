'''
Created on 03.06.2022

@author: Sascha Holzhauer
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


from fire_evacuation.model import FireEvacuation
from fire_evacuation.agent import Human

if __name__ == '__main__':
    
    #imat = {"neumann2": 0.0, "moore2": 0.5, "swnetwork": 0.0}
    
    evacuation = FireEvacuation(floor_size = 14,
            human_count = 70,
            alarm_believers_prop = 1.0,
            interact_neumann2 = 0.0,
            interact_moore2 = 0.0,
            interact_swnetwork = 0.5,
            max_speed = 2,
            seed = 3)
    
    # Run the model
    evacuation.run(100)
    
    # Store the agent memory
    memories = evacuation.get_agentmemories()
    
    for _ in range(0,15):
        evacuation = FireEvacuation(floor_size = 14,
                human_count = 70,
                alarm_believers_prop = 1.0,
                max_speed = 2,
                cooperation_mean = 0.3,
                nervousness_mean = 0.5,
                predictcrowd = True,
                agentmemories = memories,
                seed = 40)
        evacuation.run(100)
        memories = evacuation.get_agentmemories()
        
        counter = 0
        steps2escape = 0
        for agent in evacuation.schedule.agents:
            if isinstance(agent, Human):
                counter +=1
                steps2escape += agent.numsteps2escape
                
        print("Avg. steps to escape: " + str(steps2escape/counter))
        