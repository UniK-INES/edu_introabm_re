import mesa
import logging

logger = logging.getLogger("DPD")

class PDAgent(mesa.Agent):
    """Agent member of the iterated, spatial prisoner's dilemma model."""

    def __init__(self, pos, model, starting_move=None):
        """
        Create a new Prisoner's Dilemma agent.

        Args:
            pos: (x, y) tuple of the agent's position.
            model: model instance
            starting_move: If provided, determines the agent's initial state:
                           C(ooperating) or D(efecting). Otherwise, random.
        """
        super().__init__(pos, model)
        self.pos = pos
        self.score = 0
        if starting_move:
            self.move = starting_move
        else:
            self.move = self.random.choice(["C", "D"])
        self.next_move = None
        logger.debug("Initial strategy of {0:2.0f}/{1:2.0f}".format(self.pos[0], 
                                                                    self.pos[1]) + ": " +
                                                                    str(self.move))

    @property
    def isCooroperating(self):
        return self.move == "C"

    def step(self):
        """Get the best neighbor's move, and change own move accordingly
        if better than own score."""

        neighbors = self.model.grid.get_neighbors(self.pos, True, include_center=True)
        if self.model.shuffleneighbors:
            self.random.shuffle(neighbors)
        best_neighbor = max(neighbors, key=lambda a: a.score)
        self.next_move = best_neighbor.move

        if self.model.printneighbourscore:
            if self.pos == self.model.focalpos:
                logger.info("Neighbours of {0:2.0f}/{1:2.0f}".format(self.pos[0], self.pos[1]) + ": " + 
                      "".join([agent.move  + "(" + "{:4.1f}".format(agent.score) + 
                               ")-" for agent in neighbors]) + "> " + self.next_move)
        
        if self.model.printneighbourorder:
            if self.pos == self.model.focalpos:
                logger.info("Neighbours of {0:2.0f}/{1:2.0f}".format(self.pos[0], self.pos[1]) + ": " + 
                      "".join(["({:1.0f}/{:1.0f}".format(agent.pos[0], agent.pos[1]) + 
                               ") > " for agent in neighbors]))
        
        # For "Simultaneous", advance() is called by Mesa
        if self.model.schedule_type != "Simultaneous":
            self.advance()

    def advance(self):
        self.move = self.next_move
        self.score += self.increment_score()

    def increment_score(self):
        neighbors = self.model.grid.get_neighbors(self.pos, True)
        if self.model.shuffleneighbors:
            self.random.shuffle(neighbors)
        if self.model.schedule_type == "Simultaneous":
            moves = [neighbor.next_move for neighbor in neighbors]
        else:
            moves = [neighbor.move for neighbor in neighbors]

            if self.model.printneighbourscore:
                if self.pos == self.model.focalpos:
                    logger.info("Score of {0:2.0f}/{1:2.0f}: ".format(self.pos[0], self.pos[1]) + 
                          str(list(self.model.payoff[(self.move, move)] for move in moves)) + 
                          " = " + str(sum(self.model.payoff[(self.move, move)] for move in moves)))
                    
        return sum(self.model.payoff[(self.move, move)] for move in moves)
