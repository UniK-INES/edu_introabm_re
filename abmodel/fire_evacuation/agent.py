from mesa.space import Coordinate
import networkx as nx
from enum import IntEnum
from mesa import Agent
import math
import logging

from .utils import get_random_id

logger = logging.getLogger("FireEvacuation")

def get_line(start, end):
    """
    Implementation of Bresenham's Line Algorithm
    Returns a list of tuple coordinates from starting tuple to end tuple (and including them)
    """
    # Break down start and end tuples
    x1, y1 = start
    x2, y2 = end

    # Calculate differences
    diff_x = x2 - x1
    diff_y = y2 - y1

    # Check if the line is steep
    line_is_steep = abs(diff_y) > abs(diff_x)

    # If the line is steep, rotate it
    if line_is_steep:
        # Swap x and y values for each pair
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # If the start point is further along the x-axis than the end point, swap start and end
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Calculate the differences again
    diff_x = x2 - x1
    diff_y = y2 - y1

    # Calculate the error margin
    error_margin = int(diff_x / 2.0)
    step_y = 1 if y1 < y2 else -1

    # Iterate over the bounding box, generating coordinates between the start and end coordinates
    y = y1
    path = []

    for x in range(x1, x2 + 1):
        # Get a coordinate according to if x and y values were swapped
        coord = (y, x) if line_is_steep else (x, y)
        path.append(coord)  # Add it to our path
        # Deduct the absolute difference of y values from our error_margin
        error_margin -= abs(diff_y)

        # When the error margin drops below zero, increase y by the step and the error_margin by the x difference
        if error_margin < 0:
            y += step_y
            error_margin += diff_x

    # The the start and end were swapped, reverse the path
    if swapped:
        path.reverse()

    return path


"""
FLOOR STUFF
"""


class FloorObject(Agent):
    def __init__(
        self,
        traversable: bool,
        visibility: int = 2,
        model=None,
    ):
        rand_id = get_random_id(model.rng)
        super().__init__(rand_id, model)
        self.traversable = traversable
        self.visibility = visibility

    def get_position(self):
        return self.pos
    
    def __str__(self) -> str:
        return str(type(self).__name__) + str(self.pos)

    def __repr__(self):
        return self.__str__()


class Sight(FloorObject):
    def __init__(self, model):
        super().__init__(
            traversable=True, visibility=-1, model=model
        )

    def get_position(self):
        return self.pos
    

class FireExit(FloorObject):
    def __init__(self, model):
        super().__init__(
            traversable=True, visibility=6, model=model)


class Wall(FloorObject):
    def __init__(self, model):
        super().__init__(traversable=False, model=model)


class Furniture(FloorObject):
    
    def __init__(self, model):
        super().__init__(traversable=False, model=model)


class Human(Agent):
    """
    A human agent, which will attempt to escape from the grid.

    Attributes:
        ID: Unique identifier of the Agent
        Position (x,y): Position of the agent on the Grid
        Health: Health of the agent (between 0 and 1)
        ...
    """
        
    class Orientation(IntEnum):
        NORTH = 1
        EAST = 2
        SOUTH = 3
        WEST = 4

    MIN_SPEED = 0
    MAX_SPEED = 3
    
    CROWD_RADIUS = 3
    
    CROWD_RELAXATION_THRESHOLD = 0.6
    CROWD_ANXIETY_THRESHOLD = 0.8
    
    CROWD_ANXIETY_INCREASE = 0.2
    CROWD_RELAXATION_DECREASE = 0.1
    
    NERVOUSNESS_SPEEDCHANGE = 1
    NERVOUSNESS_DECREASE_HELP = 0.5
    NERVOUSNESS_INCREASE_BLOCKED = 0.2
    
    SPEED_RECOVERY_PROBABILTY = 0.15
    
    # The value the nervousness score must reach for an agent to start panic behaviour
    NERVOUSNESS_PANIC_THRESHOLD = 0.8
    NERVOUSNESS_SPEEDCHANGE_THRESHOLD = 0.6
    RANDOMWALK_PROB = 0.3
    
    COOPERATIVENESS_THRESHOLD = 0.5
    COOPERATIVENESS_EXPLORATION = 0.0
    COOPERATIVENESS_CHANGE = 0.2
    
    DECISION_SPEED = "update speed"
    DECISION_COOPERATE = "cooperate"
    DECISION_PLAN_TARGET = "plan target"
    DECISION_RANDOM_WALK = "panic random walk"
        
    def __init__(self,
            unique_id,
            speed: int,
            orientation: Orientation.NORTH,
            nervousness: float,
            cooperativeness: float,
            memory: dict,
            memorysize: int,
            believes_alarm: bool,
            model,
            turnwhenblocked_prop: float,
            switches: dict = {},
            distancenoisefactor = 1.0,
            maxsight = math.inf,
            interactionmatrix = None,
        ):
        
        """
        Update visible tiles

        Parameters
        ----------
        
        unique_id: int
            agent ID
            
        pos: Coordinate
            initial agent coordinates
            
        speed : int
            number of tiles to go during a simulation step
            
        orientation: Orientation
            initial orientation of the agent (NORTH, EAST, SOUTH, WEST)
            
        nervousness: float
            value 0...1
            
        cooperativeness: float
            value 0...1
            
        believes_alarm: bool
        
        model: Model
            model
            
        switches: dict
            switches for specific features
        
        maxsight: int
            agents' sight in grid cells
            
        interactionmatrix: dict

        Returns
        -------
        None.

        """
        
        super().__init__(unique_id, model)

        ''' Human humans should not be traversable, but we allow 
        "displacement", e.g. pushing to the side'''
        self.traversable = False
        self.orientation = orientation
        self.speed: int = speed
        self.crowdradius = Human.CROWD_RADIUS
        self.nervousness = nervousness
        self.turnwhenblocked_prop = turnwhenblocked_prop

        self.maxsight = maxsight
        self.interactionmatrix = interactionmatrix
        
        self.cooperativeness = cooperativeness
        
        self.memorysize = memorysize
        self.memory = memory
        self.learn()
        
        # Boolean stating whether or not the agent believes the alarm is a real fire
        self.believes_alarm = believes_alarm
        self.turned = False  
        self.switches = switches
        self.distancenoisefactor = distancenoisefactor
        self.escaped: bool = False
        self.numsteps2escape = -1
        
        self.visible_neighborhood = set()
        self.exits = dict()
        self.humans = dict()
        self.humantohelp = None
        
        # The agent and seen location (agent, (x, y)) the agent is planning to move to
        self.planned_target: Agent = None

        self.visible_tiles: tuple[Coordinate, tuple[Agent]] = []
        self.knownExits: tuple[Coordinate] = [] 

    def learn(self):        
        if not self.memory is None:
            lastcooperativeness = self.memory[self.memory['rep'] == max(self.memory['rep'])]['cooperativeness'].iloc[0]
            
            if self.model.modelrun < self.memorysize or self.model.rngl.random() < Human.COOPERATIVENESS_EXPLORATION:
                self.cooperativeness = lastcooperativeness + Human.COOPERATIVENESS_CHANGE * self.model.rngl.uniform(-1.0,1.0)
            else:
                # determine best cooperativeness:
                bestcooperativeness = self.memory[
                        self.memory['numsteps2escape'] == min(
                            self.memory[(self.memory['rep'] > (max(self.memory['rep']) - self.memorysize))]
                            ['numsteps2escape'])]['cooperativeness'].iloc[0]                                         
                self.cooperativeness = lastcooperativeness + (bestcooperativeness - lastcooperativeness) * \
                Human.COOPERATIVENESS_CHANGE
            self.cooperativeness = min(max(0.0, self.cooperativeness), 1.0)
            
            
    def learn_fieldofvision(self):
        self.visible_neighborhood = self.explore_fieldofvision(self.orientation)
        self.humans = dict()
        
        # add agents in found cells
        for agent in self.model.grid.iter_cell_list_contents(self.visible_neighborhood):
            if isinstance(agent, FireExit):
                self.exits[agent]=None
            elif isinstance(agent, Human):
                self.humans[agent]=None


    def explore_fieldofvision(self, orientation):

        visible_neighborhood = list()

        # gather cells in a 90° angle in the human's direction:
        if orientation == Human.Orientation.NORTH:
            startx = stopx = self.pos[0]
            for y in range(self.pos[1] + 1, min(self.model.grid.height, self.pos[1]+ self.maxsight)):
                startx = max(startx-1, 0)
                stopx = min(stopx + 1, self.model.grid.width)
                for x in range(startx, stopx):
                    visible_neighborhood.append((x,y))
        elif orientation == Human.Orientation.SOUTH:
            startx = stopx = self.pos[0]
            for y in range(self.pos[1] - 1, max(-1, self.pos[1] - self.maxsight), -1):
                startx = max(startx-1, 0)
                stopx = min(stopx + 1, self.model.grid.width)
                for x in range(startx, stopx):
                    visible_neighborhood.append((x,y))
        elif orientation == Human.Orientation.WEST:
            starty = stopy = self.pos[1]
            for x in range(self.pos[0] - 1, max(-1, self.pos[0] - self.maxsight), -1):
                starty = max(starty-1, 0)
                stopy = min(stopy + 1, self.model.grid.height)
                for y in range(starty, stopy):
                    visible_neighborhood.append((x,y))                          
        elif orientation == Human.Orientation.EAST:
            starty = stopy = self.pos[1]
            for x in range(self.pos[0] + 1, min(self.model.grid.width, self.pos[0] + self.maxsight)):
                starty = max(starty-1, 0)
                stopy = min(stopy + 1, self.model.grid.height)
                for y in range(starty, stopy):
                    visible_neighborhood.append((x,y))
        return visible_neighborhood

  
    def getEuclideanDistance(self, pos):
        return math.sqrt(abs(self.pos[0] - pos[0])**2 + abs(self.pos[1] - pos[1])**2)

      
    def cooperate(self):
        # find close-by human in need of help
        # criteria: speed = 0, not believing alarm, no exit target
        
        if len(self.humans) > 0:
            distance = math.inf
            closebyhuman = None
            for human in self.humans.keys():
                if human.speed == 0 or human.believes_alarm == False or len(human.exits) == 0:
                    curdist = self.getEuclideanDistance(human.pos)
                    if curdist < distance:
                        distance = curdist
                        closebyhuman = human
            
            if not closebyhuman == None:
                self.planned_target = closebyhuman
                self.humantohelp = closebyhuman
    

        #Perform turning of an agent
        elif(self.model.human_count - self.model.get_num_escaped(self.model)) > 1:
            self.turn()
    
    def turn(self):
        """
        Perform turning of an agent
        
        If switch 'PREDICT_CROWD' is on, considers crowds such that the agent
        turns away from crowds.
        """
        
        if 'PREDICT_CROWD' in self.switches and self.switches['PREDICT_CROWD']:
            # predict escape time
            minNumHumans = math.inf
            newOrientation = None
            
            for o in Human.Orientation:
                counter = 0
                for agent in self.model.grid.iter_cell_list_contents(self.explore_fieldofvision(o)):
                    if isinstance(agent, Human):
                        counter +=1
                if counter < minNumHumans:
                    minNumHumans = counter
                    newOrientation = o
        else:
            newOrientation = self.orientation
                
        # check whether the orientation is new and turn randomly
        while self.orientation == newOrientation:
            newOrientation = Human.Orientation(self.orientation % 4 + 1 )
        self.orientation = newOrientation
        self.turned = True
        self.model.increment_decision_count(self.model.COUNTER_TURN)


    def get_random_target(self, allow_visited=True):
        """
        Choose random tile

        Parameters
        ----------
        allow_visited : bool, optional
            The default is True.

        Returns
        -------
        None.

        """
        # exclude walls!
        x = self.model.rng.integers(2, self.model.grid.width - 2)
        y = self.model.rng.integers(2, self.model.grid.height - 2)
        self.planned_target = Agent(get_random_id(self.model.rng), self.model)
        self.planned_target.pos = (x,y)


    def attempt_exit_plan(self):
        """
        Finds a target to exit

        Returns
        -------
        None.

        """
        self.planned_target = None

        if len(self.exits) > 0:
            if len(self.exits) > 1:  
                # If there is more than one exit known
                best_distance = None
                for exitdoor in self.exits.keys():
                    # Let's use Bresenham's to find the 'closest' exit
                    if 'DISTANCE_NOISE' in self.switches and self.switches['DISTANCE_NOISE']:
                        # implement noise to the distance perception
                        length = len(get_line(self.pos, exitdoor.pos)) * self.distancenoisefactor \
                            * self.model.rng.normal(loc=1.0, scale=1.5) 
                    else:
                        length = len(get_line(self.pos, exitdoor.pos))
                    if not best_distance or length < best_distance:
                        best_distance = length
                        self.planned_target = exitdoor

            else:
                self.planned_target = list(self.exits.keys())[0]

        elif self.turned == False:
            # If there's no fire-escape in sight, turn around
            self.turn()            


    def get_next_location(self, path):
        """
        Extract the path and target for the next tick.

        Parameters
        ----------
        path : tuple
            currently followed path.

        Raises
        ------
        Exception
            Failure when determining next location.

        Returns
        -------
        next_location : pos
            Next location to end at.
        next_path : tuple
            Path to next location.

        """
        path_length = len(path)

        try:
            if path_length <= self.speed:
                next_location = path[path_length - 1]
            else:
                next_location = path[self.speed]

            next_path = []
            for location in path:
                next_path.append(location)
                if location == next_location:
                    break

            return (next_location, next_path)
        except Exception as e:
            raise Exception(
                f"Failed to get next location: {e}\nPath: {path},\nlen: {path_length},\nSpeed: {self.speed}"
            )

    def get_path(self, graph, target, include_target=True) -> list[Coordinate]:
        """
        Get path to target from graph

        Parameters
        ----------
        graph : nx graph
            graph of traversable ways over the floor plan.
        target : tile
            target tile
        include_target : bool, optional
            The default is True.

        Raises
        ------
        Exception
            DESCRIPTION.
        e
            DESCRIPTION.

        Returns
        -------
        list[Coordinate]
            an empty path if no path can be found

        """
        path = []
        visible_tiles_pos = [pos for pos, _ in self.visible_neighborhood]

        try:
            path = nx.shortest_path(graph, self.pos, target)
            if not include_target:
                del path[
                    -1
                ]  # We don't want the target included in the path, so delete the last element

            return list(path)
        except nx.exception.NodeNotFound as e:
            graph_nodes = graph.nodes()

            if target not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(target)
                logger.warn(f"Target node not found! Expected {target}, with contents {contents}")
                return path
            elif self.pos not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(self.pos)
                raise Exception(
                    f"Current position not found!\nPosition: {self.pos},\nContents: {contents}"
                )
            else:
                raise e

        except nx.exception.NetworkXNoPath as e:
            # print(f"No path between nodes! ({self.pos} -> {target})")
            return path


    def location_is_traversable(self, pos) -> bool:
        if not self.model.grid.is_cell_empty(pos):
            contents = self.model.grid.get_cell_list_contents(pos)
            for agent in contents:
                if not agent.traversable:
                    return False

        return True


    def push_human_agent(self, agent):
        """
        Pushes the agent to a neighbouring tile

        Parameters
        ----------
        agent
            agent to push.

        Returns
        -------
        None.

        """
        neighborhood = self.model.grid.get_neighborhood(
            agent.get_position(),
            moore=True,
            include_center=False,
            radius=1,
        )
        traversable_neighborhood = [
            neighbor_pos
            for neighbor_pos in neighborhood
            if self.location_is_traversable(neighbor_pos)
        ]

        if len(traversable_neighborhood) > 0:
            # push the human agent to a random traversable position
            i = self.model.rng.choice(len(traversable_neighborhood))
            push_pos = traversable_neighborhood[i]
            self.model.grid.move_agent(agent, push_pos)

            agent.nervousness += 0.1


    def getCrowdLevel(self):
        agentcounter = 0 
        neighbourhood = self.model.grid.get_neighbors(self.pos, moore=True,
                radius = self.crowdradius)   
        for agent in neighbourhood:
            if isinstance(agent, Human):
                agentcounter +=1
        return agentcounter / len(self.model.grid.get_neighborhood(self.pos, moore=True,
                radius = self.crowdradius))    


    def update_nervousness(self):
        crowdlevel = self.getCrowdLevel()
        if crowdlevel > Human.CROWD_ANXIETY_THRESHOLD:
            self.nervousness += Human.CROWD_ANXIETY_INCREASE
        elif crowdlevel < Human.CROWD_RELAXATION_THRESHOLD:
            self.nervousness -= Human.CROWD_RELAXATION_DECREASE
        self.nervousness = min(max(0.0, self.nervousness), 1.0) 

            
    def move_toward_target(self):
        next_location: Coordinate = None
        pruned_edges = set()
        graph = self.model.graph

        while self.planned_target.pos and not next_location:
            path = self.get_path(graph, self.planned_target.pos)
            
            if isinstance(self.planned_target, Human):
                # to help a human, the agent needs to be next to the human
                path = path[0:-1]

            if len(path) > 0:
                next_location, _ = self.get_next_location(path)

                if next_location == self.pos:
                    continue

                if next_location == None:
                    raise Exception("Next location can't be none")

                # Test the next location to see if we can move there
                if self.location_is_traversable(next_location):
                    # Move normally
                    self.previous_pos = self.pos
                    self.model.grid.move_agent(self, next_location)
                    
                elif self.pos == path[-1]:
                    # The human reached their target!
                   
                    self.planned_target = None
                    break

                else:
                    # We want to move here but it's blocked

                    # check if the location is blocked due to a Human agent
                    pushed = False
                    contents = self.model.grid.get_cell_list_contents(next_location)
                    for agent in contents:
                        # Test the panic value to see if this agent "pushes" the 
                        # blocking agent aside
                        if isinstance(agent, Human):
                            
                            if self.nervousness >= Human.NERVOUSNESS_PANIC_THRESHOLD:
                                # push the agent and then move to the next_location
                                self.push_human_agent(agent)
                                self.previous_pos = self.pos
                                self.model.grid.move_agent(self, next_location)
                                pushed = True
                                break
                            elif self.model.rng.random() < self.turnwhenblocked_prop:
                                self.turn()
                                break
                    if self.turned:
                        break
                    if pushed:
                        continue

                    # Remove the next location from the temporary graph so we 
                    # can try pathing again without it
                    edges = graph.edges(next_location)
                    pruned_edges.update(edges)
                    graph.remove_node(next_location)

                    # Reset planned_target if the next location was the end of the path
                    if next_location == path[-1]:
                        next_location = None
                        self.planned_target = None
                        break
                    else:
                        next_location = None

            else:  # No path is possible, so drop the target
                self.planned_target = None
                self.nervousness += Human.NERVOUSNESS_INCREASE_BLOCKED
                break

        if len(pruned_edges) > 0:
            # TODO does not seem to be necessary, as graph is not used after this in this function
            # Add back the edges we removed when removing any non-traversable nodes 
            # from the global graph, because they may be traversable again next step
            graph.add_edges_from(list(pruned_edges))

    def help(self):
        if self.humantohelp != None:
            if self.humantohelp.escaped:
                self.humantohelp = None
                self.planned_target = None
            # reached human to help?
            elif self.humantohelp in self.model.grid.get_neighbors(self.pos, moore=True):
                self.humantohelp.nervousness -= Human.NERVOUSNESS_DECREASE_HELP
                self.humantohelp.nervousness = min(max(0.0, self.humantohelp.nervousness), 1.0) 
                if self.humantohelp.speed == 0:
                    self.humantohelp.speed = 1
                elif not self.humantohelp.believes_alarm:
                    self.humantohelp.believes_alarm = True
                elif len(self.exits) > 0:
                    self.humantohelp.exits = self.exits
                self.humantohelp = None
                self.planned_target = None
                
    def propagate(self):
        if not self.interactionmatrix is None:
            if not self.interactionmatrix["moore"] is None and self.interactionmatrix["moore"] > 0:
                for other in self.model.grid.get_neighbors(self.pos, moore=True, radius=1):
                    if isinstance(other, Human):
                        if self.model.rng_propagate.random() < self.interactionmatrix["moore"]:
                            other.believes_alarm = True
            
            if not self.interactionmatrix["neumann"] is None and self.interactionmatrix["neumann"] > 0:
                for other in self.model.grid.get_neighbors(self.pos, moore=False, radius=1):
                    if isinstance(other, Human):
                        if self.model.rng_propagate.random() < self.interactionmatrix["neumann"]:
                            other.believes_alarm = True
        
            if not self.interactionmatrix["swnetwork"] is None and self.interactionmatrix["swnetwork"] > 0:
                for other in self.model.net.get_neighbors(self):
                    if isinstance(other, Human):
                        if self.model.rng_propagate.random() < self.interactionmatrix["swnetwork"]:
                            other.believes_alarm = True
        
    def step(self):
        if not self.escaped and self.pos:
            self.turned = False
            
            ######################
            # Update properties
            ######################

            # update nervousness:
            self.update_nervousness()

            # update field of vision
            self.learn_fieldofvision()

            # update speed
            if self.nervousness > Human.NERVOUSNESS_SPEEDCHANGE_THRESHOLD:
                # Either slow down or accelerate in panic situation:
                self.model.increment_decision_count(Human.DECISION_SPEED) # count
                self.speed = int(min(max(Human.MIN_SPEED, 
                                         self.speed + self.model.rng.choice([-1, 1])), Human.MAX_SPEED)) 
            
            if self.speed == 0 and self.model.rng.random() < Human.SPEED_RECOVERY_PROBABILTY:
                self.speed = 1
            
            # believe in alarm with prob = 0.002
            if not self.believes_alarm:
                if 0.002 > self.model.rng.random():
                    self.believes_alarm = True
            else:
                self.propagate()
                
            ######################
            # Decide action:
            ######################

            # check panic mode
            if self.nervousness > Human.NERVOUSNESS_PANIC_THRESHOLD:
                if self.model.rng.random() < Human.RANDOMWALK_PROB:
                    logger.debug(str(self.pos) + "Random target because of panic: " + str(self.planned_target))
                    self.get_random_target()
                    self.model.increment_decision_count(Human.DECISION_RANDOM_WALK)
            
            else:        
                # check cooperation
                if self.cooperativeness > self.COOPERATIVENESS_THRESHOLD and self.humantohelp == None \
                        and (len(self.exits) > 0 
                        or self.model.COOPERATE_WO_EXIT):
                    self.cooperate()
                    self.model.increment_decision_count(Human.DECISION_COOPERATE)
                        
                # If the agent believes the alarm, attempt to plan 
                # an exit location if we haven't already and we aren't performing an action
                if not self.turned and not isinstance(self.planned_target, FireExit) and not isinstance(self.planned_target, Human):
                    if self.believes_alarm:
                        self.attempt_exit_plan()
                        self.model.increment_decision_count(Human.DECISION_PLAN_TARGET)
                        logger.debug("Human (" + str(self.pos[0]) + "/" + str(self.pos[1])+ "): Planned target: " + self.get_planned_target())


            ######################
            # Perform action:
            ######################
            
            if not self.turned:
                if self.planned_target == None:
                    logger.debug(str(self.pos) + ": Random target because of no other: " + str(self.get_planned_target()))
                    self.get_random_target()
                
                    
                # finally go
                self.move_toward_target()
                
                self.help()
    
                # Agent reached a fire escape, proceed to exit
                if self.pos in self.model.fire_exits.keys():
                    # record escapes through exits
                    self.model.escaped(self.pos)
                    self.escaped = True
                    self.numsteps2escape = self.model.schedule.steps
                    self.model.grid.remove_agent(self)

    def get_speed(self):
        return self.speed

    def get_position(self):
        return self.pos
    
    def get_planned_target(self):
        if self.planned_target != None:
            return str(self.planned_target.pos)
        else:
            return "none"

    def set_believes(self, value: bool):
        if value and not self.believes_alarm:
            self.believes_alarm = value
            
    def __str__(self) -> str:
        return str(type(self).__name__) + str(self.pos)

    def __repr__(self):
        return self.__str__()
            
            
# Add the new Facilitator class here!

class Facilitator(Human):
    """
    A facilitator agent, which is more experiences and less likely to get nervous.

    Attributes:
        ID: Unique identifier of the Agent
        Position (x,y): Position of the agent on the Grid
        Health: Health of the agent (between 0 and 1)
        ...
    """
    
    CROWD_RELAXATION_THRESHOLD = Human.CROWD_RELAXATION_THRESHOLD + 0.1
    CROWD_ANXIETY_THRESHOLD = Human.CROWD_RELAXATION_THRESHOLD + 0.2
    
    CROWD_ANXIETY_INCREASE = 0.1
    CROWD_RELAXATION_DECREASE = 0.2


    def __init__(self,
            unique_id,
            speed: int,
            orientation: Human.Orientation.NORTH,
            nervousness: float,
            cooperativeness: float,
            memory,
            memorysize,
            turnwhenblocked_prop,
            model,
            switches = {},
            believes_alarm: bool = True,
            distancenoisefactor = 1.0,
            maxsight = math.inf,
            interactionmatrix = None,
        ):
        
        """
        Update visible tiles

        Parameters
        ----------
        
        unique_id: int
            agent ID
            
        pos: Coordinate
            initial agent coordinates
            
        speed : int
            number of tiles to go during a simulation step
            
        pos: Coordinate
            initial agent coordinates
            
        speed : int
            number of tiles to go during a simulation step
            
        orientation: Orientation
            initial orientation of the agent (NORTH, EAST, SOUTH, WEST)
            
        nervousness: float
            value 0...1
            
        cooperativeness: float
            value 0...1
            
        believes_alarm: bool
        
        model: Model
            model

        Returns
        -------
        None.

        """
        
        super().__init__(
            unique_id = unique_id,
            speed = speed,
            orientation = orientation,
            nervousness = nervousness,
            cooperativeness = cooperativeness,
            memory = memory,
            memorysize = memorysize,
            believes_alarm = True,
            turnwhenblocked_prop = turnwhenblocked_prop,
            model = model,
            switches = switches,
            distancenoisefactor = distancenoisefactor,
            maxsight = maxsight,
            interactionmatrix = interactionmatrix,
        )

    def update_nervousness(self):
        crowdlevel = self.getCrowdLevel()
        if crowdlevel > Facilitator.CROWD_ANXIETY_THRESHOLD:
            self.nervousness += Facilitator.CROWD_ANXIETY_INCREASE
        elif crowdlevel < Facilitator.CROWD_RELAXATION_THRESHOLD:
            self.nervousness -= Facilitator.CROWD_RELAXATION_DECREASE
        self.nervousness = min(max(0.0, self.nervousness), 1.0)
        
        