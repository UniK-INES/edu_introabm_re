from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import FireEvacuation
from .agent import FireExit, Wall, Human, Sight, Door, Facilitator


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
    elif type(agent) is Facilitator:
        portrayal["Shape"] = "fire_evacuation/resources/facilitator.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
        
    elif type(agent) is FireExit:
        portrayal["Shape"] = "fire_evacuation/resources/fire_exit.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Door:
        portrayal["Shape"] = "fire_evacuation/resources/door.png"
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

# initial extent that will be changed on reset (?)
canvas_element = CanvasGrid(fire_evacuation_portrayal, 30, 30, 700, 700)

# Define the charts on our web interface visualisation
status_chart = ChartModule(
    [
        {"Label": "Alive in room", "Color": "blue"},
        {"Label": "NumEscaped", "Color": "green"},
    ]
)

mobility_chart = ChartModule(
    [
        {"Label": "AvgNervousness", "Color": "red"},
        {"Label": "AvgSpeed", "Color": "green"},
    ]
)


# Specify the parameters changeable by the user, in the web interface
model_params = {
    "seed": UserSettableParameter(
        "number", "Random seed", value=1
    ),
    "floor_size": UserSettableParameter(
        "slider", "Room size (edge)", value=12, min_value=5, max_value=30, step=1
    ),
    "human_count": UserSettableParameter(
        "slider", "Number Of Human Agents", value=80, min_value=1, max_value=500, step=5
    ),
    "max_speed": UserSettableParameter(
        "slider", "Maximum Speed of agents", value=2, min_value=1, max_value=5, step=1
    ),
    "alarm_believers_prop": UserSettableParameter(
        "slider", "Proportion of Alarm Believers", value=1.0, min_value=0.0, max_value=1.0, step=0.05
    ),
    "cooperation_mean": UserSettableParameter(
        "slider", "Mean Cooperation", value=0.3, min_value=0, max_value=1, step=0.01
    ),
    "nervousness_mean": UserSettableParameter(
        "slider", "Mean Nervousness", value=0.3, min_value=0, max_value=1, step=0.01
    ),
        
    "facilitators_percentage": UserSettableParameter(
        "slider", "Percentage of facilitators", value=10, min_value=0, max_value=100, step=1
    ),
}

# Start the visual server with the model
server = ModularServer(
    FireEvacuation,
    [canvas_element, status_chart, mobility_chart],
    "Room Evacuation",
    model_params
)
