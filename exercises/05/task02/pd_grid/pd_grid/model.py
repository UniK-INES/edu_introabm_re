import mesa
import logging
from .agent import PDAgent

logger = logging.getLogger("DPD")

class PdGrid(mesa.Model):
    """Model class for iterated, spatial prisoner's dilemma model."""

    schedule_types = {
        "Sequential": mesa.time.BaseScheduler,
        "Random": mesa.time.RandomActivation,
        "Simultaneous": mesa.time.SimultaneousActivation,
    }

    # This dictionary holds the payoff for this agent,
    # keyed on: (my_move, other_move)

    payoff = {("C", "C"): 1, ("C", "D"): 0, ("D", "C"): 1.6, ("D", "D"): 0}

    def __init__(
        self, width=50, height=50, schedule_type="Random", payoffs=None, seed=None,
        printneighbourscore = False, printneighbourorder = False, initscores = False,
        shuffleneighbors = False, torus=True,
        focalpos = (0,0)
    ):
        """
        Create a new Spatial Prisoners' Dilemma Model.

        Args:
            width, height: Grid size. There will be one agent per grid cell.
            schedule_type: Can be "Sequential", "Random", or "Simultaneous".
                           Determines the agent activation regime.
            payoffs: (optional) Dictionary of (move, neighbor_move) payoffs.
            seed: seed for random number generation.
            printneighbourscore: If True, prints score and move information about each agents' neighbours.
            printneighbourorder: If True, print the neighbours coordinates in the order of playing
            focalpos: the agent whose neighbours are inspected
            playfirst:select best strategy after playing the game
        """
        super().__init__()
        
        self.reset_randomizer(seed)
        
        self.grid = mesa.space.SingleGrid(width, height, torus=True)
        self.schedule_type = schedule_type
        self.schedule = self.schedule_types[self.schedule_type](self)
        self.printneighbourscore = printneighbourscore
        self.printneighbourorder = printneighbourorder
        self.shuffleneighbors = shuffleneighbors
        self.focalpos = focalpos
        
        # Create agents
        for x in range(width):
            for y in range(height):
                agent = PDAgent((x, y), self)
                self.grid.move_agent(agent, (x, y))
                self.schedule.add(agent)
            
        # Initialise scores:
        if initscores:
            for a in self.schedule.agents:
                a.next_move = a.move
            for a in self.schedule.agents:
                a.score = a.increment_score()
            
        self.datacollector = mesa.DataCollector(
            {
                "Cooperating_Agents": lambda m: len(
                    [a for a in m.schedule.agents if a.move == "C"]
                )
            }
        )

        self.running = True
        self.datacollector.collect(self)
        logger.info("Model initialised. ")

    def step(self):
        if self.printneighbourscore or self.printneighbourorder:
            logger.info("Step " + str(self.schedule.steps))
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run(self, n):
        """Run the model for n steps."""
        for _ in range(n):
            self.step()
        logger.info("Simulation finished. ")
