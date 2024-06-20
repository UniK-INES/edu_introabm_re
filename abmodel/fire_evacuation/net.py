'''
Created on 10.06.2022

@author: Sascha Holzhauer
'''

import itertools
import math
from warnings import warn

import numpy as np

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

# For Mypy
from .agent import Agent
from numbers import Real

GridContent = Optional[Agent]

class NetworkGrid:
    """Network Grid where each node contains zero or more agents."""
    
    def __init__(self, G: Any) -> None:
        self.G = G
        self.agents = dict()
        for node_id in self.G.nodes:
            G.nodes[node_id]["agent"] = list()

    def place_agent(self, agent: Agent, node_id: int) -> None:
        """Place a agent in a node."""

        self._place_agent(agent, node_id)
        self.agents[agent] = node_id

    def get_neighbors(self, node_id: int, include_center: bool = False) -> List[int]:
        """Get all adjacent nodes"""

        neighbors = list(self.G.neighbors(node_id))
        if include_center:
            neighbors.append(node_id)

        return neighbors

    def move_agent(self, agent: Agent, node_id: int) -> None:
        """Move an agent from its current node to a new node."""

        self.remove_agent(agent)
        self._place_agent(agent, node_id)
        self.agents[agent] = node_id

    def _place_agent(self, agent: Agent, node_id: int) -> None:
        """Place the agent at the correct node."""

        self.G.nodes[node_id]["agent"].append(agent)

    def remove_agent(self, agent: Agent) -> None:
        """Remove the agent from the network"""
        node_id = self.agents[agent]
        self.G.nodes[node_id]["agent"].remove(agent)
        del self.agents[agent]

    def is_cell_empty(self, node_id: int) -> bool:
        """Returns a bool of the contents of a cell."""
        return not self.G.nodes[node_id]["agent"]

    def get_cell_list_contents(self, cell_list: List[int]) -> List[GridContent]:
        """Returns the contents of a list of cells ((x,y) tuples)
        Note: this method returns a list of `Agent`'s; `None` contents are excluded.
        """
        return list(self.iter_cell_list_contents(cell_list))

    def get_all_cell_contents(self) -> List[GridContent]:
        """Returns a list of the contents of the cells
        identified in cell_list."""
        return list(self.iter_cell_list_contents(self.G))

    def iter_cell_list_contents(self, cell_list: List[int]) -> List[GridContent]:
        """Returns an iterator of the contents of the cells
        identified in cell_list (of node IDs)."""
        list_of_lists = [
            self.G.nodes[node_id]["agent"]
            for node_id in cell_list
            if not self.is_cell_empty(node_id)
        ]
        return [item for sublist in list_of_lists for item in sublist]
    