from unittest import TestCase, mock, main
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import math

from fire_evacuation.model import FireEvacuation
from fire_evacuation.agent import Human, Facilitator


class AgentTest(TestCase):
    
    def setUp(self) -> None:
        self.model = FireEvacuation(floor_size = 14,
            human_count = 10,
            alarm_believers_prop = 1.0,
            max_speed = 2
            )
                
        self.agent = Facilitator(
                unique_id = 20,
                speed = 2,
                orientation = Human.Orientation.NORTH,
                nervousness = 0.0,
                cooperativeness = 0.8,
                believes_alarm = True,
                model = self.model,
                memory = None,
                memorysize = 0,
                turnwhenblocked_prop = 0.2,
                maxsight = math.inf,
                interactionmatrix = None,
        )
        self.model.grid.place_agent(self.agent, (3,3))

    def test_cooperate(self):
        agent2help = Human(
                unique_id = 20,
                speed = 1,
                orientation = Human.Orientation.NORTH,
                nervousness = 0.7,
                cooperativeness = 0.8,
                believes_alarm = False,
                model = self.model,
                memory = None,
                memorysize = 0,
                turnwhenblocked_prop = 0.2,
                maxsight = math.inf,
                interactionmatrix = None,
        )
        self.model.grid.place_agent(agent2help, (3,2))
        agent2help.exits = None
        self.agent.exits = (7,0)
        self.agent.humantohelp = agent2help
        self.agent.help()
        
        assert agent2help.nervousness == 0.7 - Human.NERVOUSNESS_DECREASE_HELP
        assert agent2help.speed == 1
        assert agent2help.believes_alarm == True
        assert agent2help.exits == None
        
        assert self.agent.humantohelp == None
        assert self.agent.planned_target == None
                
        