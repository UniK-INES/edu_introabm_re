from unittest import TestCase, mock, main
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
import math

from fire_evacuation.model import FireEvacuation
from fire_evacuation.agent import Human, Facilitator

class AgentTest(TestCase):
    
    def setUp(self) -> None:
        print("set up the model and agent here")


    def test_update_nervousness(self):
        assert False