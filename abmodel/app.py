from mesa.visualization import JupyterViz
from fire_evacuation.model import FireEvacuation
from fire_evacuation.agent import Human, FireExit, Wall, Sight
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

# Specify the parameters changeable by the user, in the web interface
model_params = {
    # "seed":  mesa.visualization.Number
    #      name="Random seed", value=1
    # ),
    "floor_size": {
        "type": "SliderInt",
        "value": 12,
        "label": "Room size (edge)",
        "min": 5,
        "max": 30,
        "step": 1,
    },
    "human_count": {
        "type": "SliderInt",
        "value": 80,
        "label": "Number Of Human Agents",
        "min": 1,
        "max": 500,
        "step": 5,
    },
    "max_speed": {
        "type": "SliderInt",
        "value": 2,
        "label": "Maximum Speed of agents",
        "min": 1,
        "max": 5,
        "step": 1,
    },
    "alarm_believers_prop": {
        "type": "SliderFloat",
        "value": 1.0,
        "label": "Proportion of Alarm Believers",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "cooperation_mean": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "Mean Cooperation",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "nervousness_mean": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "Mean Nervousness",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
        
    ## add slider for facilitators_percentage    
}

def agent_portrayal(agent):
    size = 1
    nervousness = None
    
    if type(agent) is Human:
        nervousness = agent.nervousness
        if agent.nervousness > Human.NERVOUSNESS_PANIC_THRESHOLD:
            shape = os.path.join(current_dir, "fire_evacuation/resources/panicked_human.png")
        elif agent.humantohelp is not None:
            shape = os.path.join(current_dir, "fire_evacuation/resources/cooperating_human.png")
        else:
            shape = os.path.join(current_dir, "fire_evacuation/resources/human.png")
    if type(agent) is FireExit:
        shape = os.path.join(current_dir, "fire_evacuation/resources/fire_exit.png")
    if type(agent) is Wall:
        shape = os.path.join(current_dir, "fire_evacuation/resources/wall.png")
    if type(agent) is Sight:
        shape = os.path.join(current_dir, "fire_evacuation/resources/eye.png")
        
    return {"size": size,
            "shape": shape,
            "Nervousness": nervousness}

# Creates a visual portrayal of our model in the browser interface
def fire_evacuation_portrayal(agent):
    if agent is None:
        return

    portrayal = {}
    (x, y) = agent.get_position()
    portrayal["x"] = x
    portrayal["y"] = y

    if type(agent) is Human:
        portrayal["scale"] = 1
        portrayal["Layer"] = 8
        portrayal["Nervousness"] = agent.nervousness
        portrayal["Cooperation"] = agent.cooperativeness
        portrayal["Believes alarm"] = str(agent.believes_alarm)
        portrayal["Turned"] = agent.turned
        portrayal["Known exits"] = str(agent.exits)
        portrayal["Target"] = agent.get_planned_target()
        portrayal["Orientation"] = agent.orientation
        portrayal["Vision"] = str(agent.visible_neighborhood)
        portrayal["Speed"] = int(agent.speed)
        portrayal["ID"]= str(agent.unique_id),
        portrayal["text_color"]= "red",
        if agent.nervousness > Human.NERVOUSNESS_PANIC_THRESHOLD:
            # Panicked
            portrayal["Shape"] = "fire_evacuation/resources/panicked_human.png"
        elif agent.humantohelp is not None:
            # Helping
            portrayal["Shape"] = "fire_evacuation/resources/cooperating_human.png"
        else:
            # Normal
            portrayal["Shape"] = "fire_evacuation/resources/human.png"
            
    # add facilitator portrayal here!
    
    elif type(agent) is FireExit:
        portrayal["Shape"] = "fire_evacuation/resources/fire_exit.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Wall:
        portrayal["Shape"] = "fire_evacuation/resources/wall.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Sight:
        portrayal["Shape"] = "fire_evacuation/resources/eye.png"
        portrayal["scale"] = 0.8
        portrayal["Layer"] = 7

    return portrayal

page = JupyterViz(
    FireEvacuation,
    model_params,
    measures=["AvgNervousness", "TurnCount"],
    name="Evacuation Model",
    agent_portrayal=agent_portrayal,
    space_drawer = "default",
)

page  # noqa


