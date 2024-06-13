import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import math
import logging

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Coordinate, MultiGrid
from mesa.time import RandomActivation

from fire_evacuation.agent import Human, Facilitator, Wall, FireExit

logger = logging.getLogger("FireEvacuation")

class FireEvacuation(Model):
    
    COUNTER_TURN = "TURN"

    COOPERATE_WO_EXIT = False
    AlARM_BELIEVERS_PROB = 0.9
    TURN_WHEN_BLOCKED_PROB = 0.5
    COOPERATION_MEAN = 0.3
    NERVOUSNESS_MEAN = 0.3
    
    def __init__(
        self,
        floor_size: int,
        human_count: int,
        visualise_vision = True,
        random_spawn = True,
        alarm_believers_prop = AlARM_BELIEVERS_PROB,
        turnwhenblocked_prop = TURN_WHEN_BLOCKED_PROB,
        max_speed = 1,
        cooperation_mean = COOPERATION_MEAN,
        nervousness_mean = NERVOUSNESS_MEAN,
        predictcrowd = False,
        agentmemories: pd.DataFrame = None,
        agentmemorysize = 5,
        seed = 1,
        facilitators_percentage = 10
     ):
        """
        

        Parameters
        ----------
        floor_size : int
            size of the room excluding walls.
        human_count : int
            DESCRIPTION.
        visualise_vision : bool
            DESCRIPTION.
        random_spawn : bool
            DESCRIPTION.
        save_plots : bool
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__()
        
        if human_count > floor_size ** 2:
            raise ValueError("Number of humans to high for the room!")
 
        
        # Not necessary?! 
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.rngl = np.random.default_rng(seed)
        self.random = np.random.default_rng(seed)
        self.MAX_SPEED = max_speed
        self.COOPERATE_WO_EXIT = FireEvacuation.COOPERATE_WO_EXIT
        
        self.debug = False
        
        self.switches = {
            'PREDICT_CROWD': predictcrowd,
            }
        
        self.stepcounter = -1
        self.agentmemory = agentmemories
        
        if not agentmemories is None:
            self.modelrun = np.max(agentmemories['rep'])
        else:
            self.modelrun = -1

        # Create floorplan
        floorplan = np.full((floor_size + 2, floor_size + 2), '_')
        floorplan[(0,-1),:]='W'
        floorplan[:,(0,-1)]='W'
        floorplan[math.floor((floor_size + 2)/2),(0,-1)] = 'E'
        floorplan[(0,-1), math.floor((floor_size + 2)/2)] = 'E'
        
        # Create floorplan with thicker walls
        floorplan = np.full((floor_size + 4, floor_size + 4), '_')
        floorplan[(0,1,-2,-1),:]='W'
        floorplan[:,(0,1,-2,-1)]='W'
        floorplan[math.floor((floor_size + 4)/2),(0,-1)] = 'E'
        floorplan[(0,-1), math.floor((floor_size + 4)/2)] = 'E'
        
        floorplan[math.floor((floor_size + 4)/2),(1,-2)] = None
        floorplan[(1,-2), math.floor((floor_size + 4)/2)] = None
        
        # distribute agent positions at the south:
        for i in range(human_count):
            floorplan[2+(i % (floor_size)), 2 + math.floor(i / (floor_size))] = 'S'
        
        # Rotate the floorplan so it's interpreted as seen in the text file
        floorplan = np.rot90(floorplan, 3)

        # Init params
        self.width = floor_size + 4
        self.height = floor_size + 4
        self.human_count = human_count
        self.visualise_vision = visualise_vision

        # Set up model objects
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(floor_size + 4, floor_size + 4, torus=False)

        # Used to easily see if a location is a FireExit, since this needs to be done a lot
        self.fire_exits: dict[Coordinate, FireExit] = {}

        # If random spawn is false, spawn_pos_list will contain the list of possible 
        # spawn points according to the floorplan
        self.random_spawn = random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        self.decisioncount = dict()
        self.exitscount = dict()
        
        # Load floorplan objects
        for (x, y), value in np.ndenumerate(floorplan):
            pos: Coordinate = (x, y)

            value = str(value)
            floor_object = None
            if value == "W":
                floor_object = Wall(self)
            elif value == "E":
                floor_object = FireExit(self)
                self.fire_exits[pos] = floor_object
            elif value == "S":
                self.spawn_pos_list.append(pos)
            if floor_object:
                self.grid.place_agent(floor_object, pos)
                self.schedule.add(floor_object)

        # Create a graph of traversable routes, used by humans for pathing
        self.graph = nx.Graph()
        for agents, pos in self.grid.coord_iter():
            # If the location is empty, or there are no non-traversable objects
            if len(agents) == 0 or not any(not agent.traversable for agent in agents):
                neighbors_pos = self.grid.get_neighborhood(
                    pos, moore=True, include_center=True, radius=1
                )

                for neighbor_pos in neighbors_pos:
                    # If the neighbour position is empty, or no non-traversable 
                    # contents, add an edge
                    if self.grid.is_cell_empty(neighbor_pos) or not any(
                        not agent.traversable
                        for agent in self.grid.get_cell_list_contents(neighbor_pos)
                    ):
                        self.graph.add_edge(pos, neighbor_pos)

        # Collects statistics from our model run
        self.datacollector = DataCollector(
            {
                "NumEscaped" : lambda m: self.get_num_escaped(m),
                "AvgNervousness": lambda m: self.get_human_nervousness(m),
                "AvgSpeed": lambda m: self.get_human_speed(m),
                
                "TurnCount": lambda m: self.get_decision_count(self.COUNTER_TURN),
                # add entries for your further decision categories here
                
                "TurnCount": lambda m: self.get_decision_count(self.COUNTER_TURN),
                "UpdateSpeedCount": lambda m: self.get_decision_count(Human.DECISION_SPEED),
                "CooperateCount": lambda m: self.get_decision_count(Human.DECISION_COOPERATE),
                "PlanTargetCount": lambda m: self.get_decision_count(Human.DECISION_PLAN_TARGET),
                "RandomWalkCount": lambda m: self.get_decision_count(Human.DECISION_RANDOM_WALK),
                
                "EscapedWest": lambda m: self.get_escaped_exit(list(self.fire_exits)[0]),
                "EscapedSouth": lambda m: self.get_escaped_exit(list(self.fire_exits)[1]),
                "EscapedNorth": lambda m: self.get_escaped_exit(list(self.fire_exits)[2]),
                "EscapedEast": lambda m: self.get_escaped_exit(list(self.fire_exits)[3]),
             }
        )
        
        # Start placing human humans
        for i in range(0, self.human_count):
            if self.random_spawn:  # Place human humans randomly
                pos = tuple(self.rng.choice(tuple(self.grid.empties)))
            else:  # Place human humans at specified spawn locations
                pos = tuple(self.rng.choice(self.spawn_pos_list))

            if pos != None:
                # Create a random human
                speed = self.rng.integers(Human.MIN_SPEED, Human.MAX_SPEED + 1)

                nervousness = -1
                while nervousness < 0 or nervousness > 1:
                    nervousness = self.rng.normal(loc = nervousness_mean, scale = 0.2)
                    
                cooperativeness = -1
                while cooperativeness < 0 or cooperativeness > 1:
                    cooperativeness = self.rng.normal(cooperation_mean)

                belief_distribution = [alarm_believers_prop, 1 - alarm_believers_prop]
                believes_alarm = self.rng.choice([True, False], p=belief_distribution)

                orientation = Human.Orientation(self.rng.integers(1,5))
                
                if (not self.agentmemory is None) and (i in self.agentmemory['agent'].values):
                    memory = self.agentmemory[self.agentmemory['agent']==i]
                else:
                    memory = None
                    
                # decide here whether to add a facilitator
                if (i < math.floor(human_count*(facilitators_percentage/100.0))):
                    while nervousness < 0 or nervousness > 1:
                        nervousness = self.rng.normal(loc = nervousness_mean, scale = 0.2)
                    human = Facilitator(
                        i,
                        speed=speed,
                        orientation = orientation,
                        nervousness = nervousness,
                        cooperativeness=cooperativeness,
                        switches = self.switches,
                        model=self,
                        memory = memory,
                        memorysize = agentmemorysize,
                        turnwhenblocked_prop = turnwhenblocked_prop,
                        )
                else:
                    while nervousness < 0 or nervousness > 1:
                        nervousness = self.rng.normal(loc = nervousness_mean - 0.3, scale = 0.2)
                    human = Human(
                        i,
                        speed=speed,
                        orientation=orientation,
                        nervousness=nervousness,
                        cooperativeness=cooperativeness,
                        believes_alarm=believes_alarm,
                        turnwhenblocked_prop = turnwhenblocked_prop,
                        switches = self.switches,
                        model=self,
                        memory = memory,
                        memorysize = agentmemorysize
                    )

                self.grid.place_agent(human, pos)
                self.schedule.add(human)
            else:
                print("No tile empty for human placement!")

        self.running = True
        logger.info("Model initialised")

    def step(self):
        """
        Advance the model by one step.
        """

        logger.info("Running step " + str(self.schedule.steps))
        self.schedule.step()
        self.datacollector.collect(self)

        # If all humans escaped, stop the model and collect the results
        if self.get_num_escaped(self) == self.human_count:
            self.running = False
        
        if self.stepcounter == 0:
            self.running = False
            # final actions:
            # create agent memory
            for agent in self.schedule.agents:
                if isinstance(agent, Human):
                    data = pd.DataFrame({'rep': self.modelrun + 1,
                                                      'agent':agent.unique_id,
                                                      'cooperativeness' : agent.cooperativeness,
                                                      #'numsteps2escape': self.schedule.steps},index = [0])
                                                      'numsteps2escape': agent.numsteps2escape},index = [0])
                    if not self.agentmemory is None:
                        self.agentmemory = pd.concat([self.agentmemory, data])
                    else:
                        self.agentmemory = data
                
        if self.stepcounter >= 0:
            self.stepcounter -=1
        elif self.get_human_speed(self) == 0:
            self.stepcounter = 10 * sum(map(lambda agent : isinstance(agent, Human) and not agent.escaped, self.schedule.agents))
                
    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            if self.running or self.stepcounter >= 0:
                self.step()
            
    def get_agentmemories(self):
        return self.agentmemory
       
    @staticmethod     
    def get_human_nervousness(model):
        count = 0
        nervousness = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and not agent.escaped:
                nervousness += agent.nervousness
                count +=1
        if count == 0:
            return 0
        return nervousness/count
 
    
    @staticmethod     
    def get_human_speed(model):
        count = 0
        speed = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and not agent.escaped:
                speed += agent.speed
                count +=1
        if count == 0:
            return 0
        return speed/count


    @staticmethod
    def get_num_escaped(model):
        """
        Helper method to count escaped humans
        """
        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Human) and agent.escaped == True:
                count += 1

        return count
    

    def increment_decision_count(self, decision):
        """
        Increments the decision counter identified by decision by one.
        Used to count decision all agents do during a step or simulation run.
        
        Args:
            decision: identifier for the specific kind of decision (eg. "TURN")
        """
        if decision not in self.decisioncount:
            self.decisioncount[decision] = 0
        self.decisioncount[decision] +=1 
    

    def get_decision_count(self, decision):
        """
        Retrieve the number of performed decisions (counted when calling
        increment_decision_count(decision)) of the specified kind (decision).
        
        Args:
            decision: identifier for the specific kind of decision (eg. "TURN")
        """
        if decision not in self.decisioncount:
            return 0
        return self.decisioncount[decision]

    def escaped(self, pos):
        if pos not in self.exitscount:
            self.exitscount[pos] = 0
        self.exitscount[pos] +=1 


    def get_escaped_exit(self, pos):
        if pos not in self.exitscount.keys():
            return 0
        else:
            return self.exitscount[pos]