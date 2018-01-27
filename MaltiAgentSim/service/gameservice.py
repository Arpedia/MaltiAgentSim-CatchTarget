# -*- encoding: utf-8
from model.area import Area 
from model.agent import Agent
from model.target import Target

class GameService:

    def __init__(self):
        'SERVICE'

    def construct(self, area_shape, mergin):
        area = Area(area_shape, mergin)
        agent1 = Agent(area)
        area.update_state(agent1)
        agent2 = Agent(area)
        area.update_state(agent2)
        target = Target(area)
        area.update_state(target)

        return (area, agent1, agent2, target)


    def turn(self, area, agent1, agent2, target, epsilon):
        agent1.lookout(area)
        agent2.lookout(area)
        agent1.act(area, epsilon)
        agent2.act(area, epsilon)
        target.act(area)


    def reset(self, area, agent1, agent2, target):
        area.reset_state()
        agent1.reset_state(area)
        area.update_state(agent1)
        agent2.reset_state(area)
        area.update_state(agent2)
        target.reset_state(area)
        area.update_state(target)