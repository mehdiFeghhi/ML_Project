from part_two.HW4_Model import NSFrozenLake
from part_two.model_free import *

env = NSFrozenLake(401722136)
env.render()

Q_learning_res= Q_learning(env, lr=.1, num_episodes=2000, eps=0.4, gamma=0.9, eps_decay=0.001)


