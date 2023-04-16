from baselines.agents import phase_one
from baselines.baselines import LowPlayer
from portal.python.main import PlayAgainstGameManager, SimulatePlayGameManager
from portal.python.eval_hand import set_agent

agent = phase_one('10_190')
#agent = LowPlayer()

set_agent(agent)


SimulatePlayGameManager(
    starting_player=0,
    agent=agent,
).run()



PlayAgainstGameManager(
    starting_player=0,
    agent=agent,
    turn_wait=1
).run()
