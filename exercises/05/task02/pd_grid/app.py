from mesa.visualization import JupyterViz
from pd_grid.model import PdGrid

import os

current_dir = os.path.dirname(os.path.realpath(__file__))

model_params = {
    "height": {
        "type": "SliderInt",
        "value": 50,
        "label": "Height",
        "min": 5,
        "max": 100,
        "step": 1,
        },
    "width": {
        "type": "SliderInt",
        "value": 50,
        "label": "Width",
        "min": 5,
        "max": 100,
        "step": 1,
        },
    "schedule_type": {
        "type": "Select",
        "label": "Scheduler type",
        "value": "Random",
        "values": list(PdGrid.schedule_types.keys()),
        },
    "printneighbourorder": True,
}

def portray_pdagent(agent):
    """
    This function is registered with the visualization server to be called
    each tick to indicate how to draw the agent in its current state.
    :param agent:  the agent in the simulation
    :return: the portrayal dictionary
    """
    if agent is None:
        raise AssertionError
    return    {        
        "Shape": "rect",
        "size": 5.0,
        "Layer": 0,
        "color": "blue" if agent.isCooroperating else "red",
    }
    

page = JupyterViz(
    PdGrid,
    model_params,
    measures=["Cooperating_Agents"],
    name="Prisoner's Dilemma",
    agent_portrayal=portray_pdagent,
    space_drawer = "default",
)


